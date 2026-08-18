"""
eval_v13_tracker_ablation.py — Tracker Ablation: Raw / KF / IMM-KF  (metric-only, Tier A)
======================================================================================
[목적]
    Graph-Token MAPPO (Proposed) policy는 고정하고, 평가용 external
    TargetBeliefTracker만 3가지 variant로 교체해서 belief 추정 방식이
    tracking metric에 미치는 영향을 비교한다.

        1. raw  — RawPositionTracker   : 동적 belief 없음 (motion model 없음)
        2. kf   — CVKFTracker          : 단일 constant-velocity Kalman filter
        3. imm  — TargetBeliefTracker  : IMM-KF (isaac_env_v13.py 기존 구현, 2-model mixture)

[중요 — 두 종류의 tracker ablation]
    A. metric-only tracker comparison   (이 파일)
       - 동일한 Graph-Token policy trajectory를 사용
       - external tracker만 raw/kf/imm으로 바꿔서 metric만 다시 계산
       - 빠르지만, policy 자체가 tracker 정보를 보고 학습된 것이므로
         엄밀한 "policy ablation"은 아님 (policy obs에 들어가는 tracker는 그대로 IMM).
    B. policy-aware tracker ablation
       - policy observation에 들어가는 tracker 자체를 raw/kf/imm으로 바꿔서
         각각 따로 학습해야 함 (논문에서 "Tracker Ablation"을 강하게 주장하려면 이게 맞음)
       - 이 파일은 다루지 않음. A가 잘 동작하는 걸 먼저 확인한 후, B로 확장.

[기준 파일]
    eval_v13_ablation.py 의 구조(SCALE_CONFIGS, STAGE1_REWARDS, CSV_HEADER,
    resume 로직, _run_ext_tracker_step의 predict_all() 우선 순서,
    _run_rl_episodes_dual_obs)를 그대로 가져왔다.
    Raw/KF tracker의 state-machine(UNCOMMITTED→CANDIDATE→COMMITTED→STALE)과
    TR_P_MAX 정규화는 isaac_env_v13.TargetBeliefTracker와 동일한 threshold를
    그대로 재사용해서, "추정 방식"만 다르고 commit/stale 판정 기준은
    세 variant 모두 동일하게 유지한다 (공정한 비교를 위한 핵심 설계).

[CSV 출력]  <repository root>/eval_results/v13/tracker_ablation/
    tracker_ablation_{scale}_{raw|kf|imm}.csv  (총 9개)

[실행법]
    python eval_v13_tracker_ablation.py --n_eval 100 ^
        --ckpt_graph_token "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt"

    # 특정 tracker/scale만
    python eval_v13_tracker_ablation.py --n_eval 100 --methods raw imm --scales 4x20
"""

import os, sys, random, time, argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR   = str(REPO_ROOT / "eval_results" / "v13" / "tracker_ablation")
MAX_STEPS = 3000

SCALE_CONFIGS = {
    "4x20":  {"num_drones": 4,  "num_targets": 20, "k_active": 12},
    "8x40":  {"num_drones": 8,  "num_targets": 40, "k_active": 12},
    "12x60": {"num_drones": 12, "num_targets": 60, "k_active": 12},
}

# Graph-Token policy는 고정 — variant 비교 대상이 아니므로 ckpt 1개만 사용
# (실제 checkpoint는 repo에 포함되지 않음; training을 이 repo 안에서 실행했다면
#  자동으로 찾고, 아니면 --ckpt_graph_token 으로 명시적으로 지정한다.)
CKPT_DIR    = str(REPO_ROOT / "graph_token_mappo_v13" / "stage1" / "seed0")
CKPT_PREFIX = "graph_token_ep"

TRACKER_METHODS = ["raw", "kf", "imm"]

STAGE1_REWARDS = {
    "LAMBDA_TRACK":              0.3,
    "LAMBDA_STALE":              0.01,
    "REWARD_UNCERTAINTY_REDUCE": 1.5,
    "PENALTY_TERRAIN":          -0.05,
    "PENALTY_DRONE_COLLISION":  -1.0,
    "ANTIAIR_KILL_PENALTY":      0.0,
    "BATTERY_COST_TRACK":        0.0001,
}

# Raw tracker 전용 튜닝 상수 (motion model이 없는 baseline — velocity는 추정하지 않음)
RAW_COV_GROWTH_POS = 2.0    # 미관측 1스텝당 P[0,0]/P[1,1] 증가량 (motion model 없음 → 빠르게 불확실해짐)
RAW_COV_GROWTH_VEL = 0.2    # 미관측 1스텝당 P[2,2]/P[3,3] 증가량 (velocity는 항상 모른다고 가정)
RAW_POS_RESET_MIN  = 1.0    # 관측 시 위치 covariance 하한 (R이 너무 작아도 과신하지 않게)

# ── CSV 헤더 — eval_v13_ablation.py와 동일 (같은 분석 스크립트로 비교 가능) ──
CSV_HEADER = (
    "ep_idx,ep_steps,num_drones,num_targets,"
    "uncertainty_pct,unc_reduction_pct,initial_unc_pct,"
    "ext_mean_trP_final,ext_auc_mean_trP,"
    "ext_committed_final,ext_committed_avg,"
    "ext_stale_avg,ext_tracked_avg,ext_stale_ratio,"
    "ext_reacquire_count,"
    "ext_T_commit_50,ext_T_trP_below_06,ext_time_under_good_track,"
    "ext_trP_infantry,ext_trP_tank,ext_trP_artillery,"
    "committed_ratio,stale_ratio_norm,"
    "drone_prox_events,mean_battery,rtb_count,depleted_ep,active_at_end,"
    "prox_per_drone,prox_per_1000steps,"
    "time_s\n"
)


# ══════════════════════════════════════════════════════════════════════
# 공통 유틸 (eval_v13_ablation.py 동일)
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def set_seed_torch(seed):
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _drone_positions(env):
    return np.array([env.drone_positions[i][:2]
                     for i in range(env.num_drones)], dtype=np.float64)

def _target_positions(env):
    return np.array([env.target_positions[i][:3]
                     for i in range(env.num_targets)], dtype=np.float64)

def _write_row(out_path, line):
    while True:
        try:
            with open(out_path, "a") as f: f.write(line)
            return
        except PermissionError:
            print(f"\n  [경고] '{os.path.basename(out_path)}' 가 열려 있습니다.")
            input("  >>> 파일을 닫은 후 Enter: ")

def _get_done_eps(out_path):
    if not os.path.exists(out_path): return set()
    done = set()
    try:
        with open(out_path, "r") as f: lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if parts:
                try: done.add(int(parts[0]))
                except ValueError: pass
    except Exception: pass
    return done

def find_latest_ckpt(folder, prefix):
    if not os.path.isdir(folder): return None
    pts = [f for f in os.listdir(folder)
           if f.startswith(prefix) and f.endswith(".pt")]
    if not pts: return None
    def _ep(f):
        try: return int(f.replace(prefix, "").replace(".pt", ""))
        except: return -1
    pts.sort(key=_ep)
    path = os.path.join(folder, pts[-1])
    print(f"  [자동 탐색] {path}")
    return path

def _make_env_v13(num_drones, num_targets, k_active):
    import isaac_env_v13 as env_module

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k):
            setattr(env_module, k, v)

    env = env_module.DroneSwarmEnv(
        randomize_terrain=False,
        terrain_seed=42,
        num_drones=num_drones,
        num_targets=num_targets,
        k_active=k_active,
    )
    return env, env_module


# ── 외부 Tracker 1스텝 갱신 — tracker variant에 무관 (공용 인터페이스만 사용) ──
def _run_ext_tracker_step(tracker, env):
    """
    predict_all() → observation update → update_target_states() 순서.
    raw/kf/imm 모두 동일한 메서드 시그니처를 구현하므로 수정 없이 그대로 동작한다.
    """
    import isaac_env_v13
    R_TRACK = isaac_env_v13.R_TRACK_M

    base_env = getattr(env, "env", env)
    tracker.predict_all()

    drone_xy = _drone_positions(base_env)
    tgt_pos  = _target_positions(base_env)
    camo_arr = getattr(base_env, "target_is_camouflaged",
                       [False] * base_env.num_targets)

    for k in range(base_env.num_targets):
        dists = np.linalg.norm(drone_xy - tgt_pos[k, :2], axis=1)
        n_obs = int(np.sum(dists < R_TRACK))
        if n_obs > 0:
            is_camo = bool(camo_arr[k])
            tracker.update(k, tgt_pos[k, :2], is_camo, n_obs)
        else:
            tracker.reset_dwell(k)
            # track_age는 predict_all() 내부에서 증가 — 여기서 수동 증가 불필요
    tracker.update_target_states()

def _collect_ext(tracker, step, n_targets,
                 trP_hist, commit_hist, stale_hist, tracked_hist,
                 T_commit_50, T_trP_below, GOOD=0.6):
    import isaac_env_v13
    TargetState = isaac_env_v13.TargetState
    trP_vals = [tracker.get_tr_P_norm(k) for k in range(n_targets)]
    mean_trP = float(np.mean(trP_vals))
    states   = tracker.target_states
    comm_n   = int(np.sum(states == TargetState.COMMITTED))
    stal_n   = int(np.sum(states == TargetState.STALE))
    track_n  = int(np.sum(states != TargetState.UNCOMMITTED))
    trP_hist.append(mean_trP); commit_hist.append(comm_n)
    stale_hist.append(stal_n); tracked_hist.append(track_n)
    if T_commit_50[0] is None and comm_n >= n_targets * 0.5: T_commit_50[0] = step
    if T_trP_below[0] is None and mean_trP < GOOD:           T_trP_below[0] = step

def _summarize_ext(tracker, trP_hist, commit_hist, stale_hist, tracked_hist,
                   T_commit_50, T_trP_below, target_types, n_targets, GOOD=0.6):
    import isaac_env_v13
    TargetState = isaac_env_v13.TargetState
    TargetType  = isaac_env_v13.TargetType
    n  = len(trP_hist)
    ta = np.array(trP_hist); ca = np.array(commit_hist)
    sa = np.array(stale_hist); tra = np.array(tracked_hist)

    def _tm(ttype):
        idxs = [i for i, t in enumerate(target_types) if int(t) == ttype]
        return float(np.mean([tracker.get_tr_P_norm(i) for i in idxs])) if idxs else float("nan")

    avg_trk = float(np.mean(tra)) if np.any(tra > 0) else 1.0
    avg_st  = float(np.mean(sa))
    reacq   = int(np.sum((tracker.prev_states == TargetState.STALE) &
                          (tracker.target_states == TargetState.COMMITTED)))
    return {
        "ext_mean_trP_final":        float(ta[-1]) if n else 1.0,
        "ext_auc_mean_trP":          float(np.mean(ta)) if n else 1.0,
        "ext_committed_final":       int(ca[-1]) if n else 0,
        "ext_committed_avg":         float(np.mean(ca)) if n else 0.0,
        "ext_stale_avg":             avg_st,
        "ext_tracked_avg":           avg_trk,
        "ext_stale_ratio":           avg_st / max(avg_trk, 1.0),
        "ext_reacquire_count":       reacq,
        "ext_T_commit_50":           T_commit_50[0] if T_commit_50[0] is not None else n,
        "ext_T_trP_below_06":        T_trP_below[0] if T_trP_below[0] is not None else n,
        "ext_time_under_good_track": float(np.mean(ta < GOOD)) if n else 0.0,
        "ext_trP_infantry":          _tm(int(TargetType.INFANTRY)),
        "ext_trP_tank":              _tm(int(TargetType.TANK)),
        "ext_trP_artillery":         _tm(int(TargetType.ARTILLERY)),
    }

def _make_csv_line(ep_idx, ep_steps, num_drones, num_targets,
                   init_unc, u_pct, drone_prox, battery,
                   rtb_count, depleted, active, ext, elapsed):
    unc_red          = (init_unc - u_pct) / max(init_unc, 1e-6) * 100.0
    committed_ratio  = ext["ext_committed_avg"] / max(num_targets, 1)
    stale_ratio_norm = ext["ext_stale_avg"]     / max(num_targets, 1)
    prox_per_drone   = drone_prox / max(num_drones, 1)
    prox_per_1000    = drone_prox / max(ep_steps / 1000.0, 1e-6)
    return (
        f"{ep_idx},{ep_steps},{num_drones},{num_targets},"
        f"{u_pct:.2f},{unc_red:.2f},{init_unc:.2f},"
        f"{ext['ext_mean_trP_final']:.4f},{ext['ext_auc_mean_trP']:.4f},"
        f"{ext['ext_committed_final']},{ext['ext_committed_avg']:.2f},"
        f"{ext['ext_stale_avg']:.2f},{ext['ext_tracked_avg']:.2f},"
        f"{ext['ext_stale_ratio']:.4f},{ext['ext_reacquire_count']},"
        f"{ext['ext_T_commit_50']},{ext['ext_T_trP_below_06']},"
        f"{ext['ext_time_under_good_track']:.4f},"
        f"{ext['ext_trP_infantry']:.4f},{ext['ext_trP_tank']:.4f},"
        f"{ext['ext_trP_artillery']:.4f},"
        f"{committed_ratio:.4f},{stale_ratio_norm:.4f},"
        f"{drone_prox},{battery:.4f},{rtb_count},{depleted},{active},"
        f"{prox_per_drone:.4f},{prox_per_1000:.4f},"
        f"{elapsed:.1f}\n"
    )


# ══════════════════════════════════════════════════════════════════════
# Tracker variant 클래스 — Raw / KF
# (IMM은 새로 만들지 않고 isaac_env_v13.TargetBeliefTracker를 그대로 사용)
# ══════════════════════════════════════════════════════════════════════

class _BaseExtTracker:
    """
    Raw / KF tracker 공용 베이스.
    isaac_env_v13.TargetBeliefTracker와 동일한 state-machine
    (UNCOMMITTED→CANDIDATE→COMMITTED→STALE), 동일한 TR_P_MAX 정규화,
    동일한 dwell 기반 R 계산을 재사용한다.
    → 세 variant의 차이가 "추정 방식"에서만 나오게 만드는 핵심 장치.
    """

    def __init__(self, n_targets, target_types, init_positions=None):
        import isaac_env_v13 as _ie
        self._ie          = _ie
        self.n_targets    = n_targets
        self.num_targets  = n_targets
        self.target_types = target_types

        self.mu = np.zeros((n_targets, 4), dtype=np.float64)
        self.P  = np.zeros((n_targets, 4, 4), dtype=np.float64)
        self.track_age   = np.zeros(n_targets, dtype=np.int32)
        self.dwell_count = np.zeros(n_targets, dtype=np.int32)
        self.target_states = np.zeros(n_targets, dtype=np.int32)
        self.prev_states    = np.zeros(n_targets, dtype=np.int32)

        self.reset(init_positions)

    def reset(self, target_positions_init=None):
        _ie = self._ie
        for k in range(self.n_targets):
            t = self.target_types[k]
            cx, cy = _ie.TARGET_SPAWN_PRIOR.get(t, (30.0, 0.0))
            self.mu[k, 0] = cx + np.random.randn() * np.sqrt(_ie.P_INIT_POS) * 0.3
            self.mu[k, 1] = cy + np.random.randn() * np.sqrt(_ie.P_INIT_POS) * 0.3
            self.mu[k, 2:] = 0.0
            self.P[k] = np.diag([_ie.P_INIT_POS, _ie.P_INIT_POS,
                                 _ie.P_INIT_VEL, _ie.P_INIT_VEL])
        self.track_age[:]   = _ie.T_STALE + 1
        self.dwell_count[:] = 0
        self.target_states[:] = _ie.TargetState.UNCOMMITTED
        self.prev_states[:]   = _ie.TargetState.UNCOMMITTED

    def reset_dwell(self, k):
        self.dwell_count[k] = 0

    def get_tr_P(self, k):
        return float(np.trace(self.P[k]))

    def get_tr_P_norm(self, k):
        return float(np.clip(np.trace(self.P[k]) / self._ie.TR_P_MAX, 0.0, 1.0))

    def get_age_norm(self, k):
        return float(np.clip(self.track_age[k] / (self._ie.T_STALE * 2), 0.0, 1.0))

    def _compute_R(self, k, n_obs, is_camo):
        _ie = self._ie
        dc = self.dwell_count[k]
        if is_camo:
            r_base = _ie.EKF_R_CAMO
        elif dc >= _ie.DWELL_FULL:
            r_base = _ie.EKF_R_DWELL
        else:
            r_base = _ie.EKF_R_INIT * (_ie.DWELL_DECAY ** dc)
        r_eff = r_base / max(1, n_obs)
        return np.diag([r_eff, r_eff]).astype(np.float64)

    def update_target_states(self):
        _ie = self._ie
        TargetState = _ie.TargetState
        self.prev_states[:] = self.target_states[:]
        for k in range(self.n_targets):
            trP = self.get_tr_P_norm(k)
            age = self.track_age[k]
            dw  = self.dwell_count[k]
            st  = self.target_states[k]
            if st == TargetState.UNCOMMITTED:
                if trP < _ie.TR_P_CANDIDATE_THRESH and age == 0:
                    self.target_states[k] = TargetState.CANDIDATE
            elif st == TargetState.CANDIDATE:
                if trP < _ie.TR_P_COMMITTED_THRESH and dw >= _ie.DWELL_FOR_COMMIT:
                    self.target_states[k] = TargetState.COMMITTED
                elif age > _ie.T_STALE:
                    self.target_states[k] = TargetState.UNCOMMITTED
            elif st == TargetState.COMMITTED:
                if age > _ie.T_STALE:
                    self.target_states[k] = TargetState.STALE
            elif st == TargetState.STALE:
                if age == 0 and trP < _ie.TR_P_RECOVER_THRESH:
                    self.target_states[k] = TargetState.COMMITTED


class RawPositionTracker(_BaseExtTracker):
    """
    1. Raw / no dynamic belief.
    motion model 없음 — 관측되면 최신 측정값을 그대로 신뢰(snap), 안 보이면
    마지막 위치를 고정한 채 covariance만 고정 비율로 증가시킨다.
    velocity는 항상 0 (속도 추정 자체를 하지 않음).
    """

    def predict_all(self):
        for k in range(self.n_targets):
            self.track_age[k] = min(self.track_age[k] + 1, 9999)

            # no-motion-model baseline: uncertainty grows quickly during missed observations
            self.P[k, 0, 0] += RAW_COV_GROWTH_POS
            self.P[k, 1, 1] += RAW_COV_GROWTH_POS

            # velocity is not estimated by the raw tracker, so velocity uncertainty should remain high
            self.P[k, 2, 2] += RAW_COV_GROWTH_VEL
            self.P[k, 3, 3] += RAW_COV_GROWTH_VEL

            self.P[k] = np.clip(self.P[k], -1e4, 1e4)

    def update(self, k, z_xy, is_camo=False, n_obs=1):
        # raw도 motion model이 없을 뿐, 센서 노이즈는 동일하게 받는다
        # (KF/IMM과 같은 R 모델로 노이즈를 주입해 "추정 방식"만 다르게 비교)
        R = self._compute_R(k, n_obs, is_camo)
        r_noise = np.sqrt(np.diag(R))
        z_noisy = np.asarray(z_xy, dtype=np.float64) + np.random.randn(2) * r_noise

        self.mu[k, 0:2] = z_noisy
        self.mu[k, 2:4] = 0.0

        # 위치는 최신 관측으로 갱신하지만, 속도는 모른다고 본다 (motion model 없음)
        pos_var = max(float(R[0, 0]), RAW_POS_RESET_MIN)
        self.P[k] = np.diag([
            pos_var,
            pos_var,
            self._ie.P_INIT_VEL,
            self._ie.P_INIT_VEL,
        ]).astype(np.float64)

        self.dwell_count[k] = min(self.dwell_count[k] + 1, self._ie.DWELL_FULL)
        self.track_age[k] = 0


class CVKFTracker(_BaseExtTracker):
    """
    2. KF — 단일 Constant-Velocity Kalman Filter (IMM 없음, motion model 1개).
    isaac_env_v13.py에 남아있는 legacy 단일-EKF 상수(EKF_Q_POS/VEL,
    EKF_Q_POS_TANK/VEL_TANK, EKF_Q_POS_STATIC/VEL_STATIC)를 그대로 사용해서
    P_INIT_POS/VEL, TR_P_MAX와 동일한 스케일로 맞춘다.
    ANTI_AIR는 IMM 버전과 동일하게 static model 사용 (그 부분은 ablation 대상이 아님).
    TANK/INFANTRY/ARTILLERY는 단일 CV 모델 + (타입별) 고정 Q.
    → 논문 설명: "KF uses a single constant-velocity model, whereas IMM-KF
       maintains multiple motion hypotheses."
    """

    @staticmethod
    def _make_F_cv(dt=1.0):
        return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)

    @staticmethod
    def _make_F_static():
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=np.float64)

    def _get_kf_config(self, k):
        _ie = self._ie
        t = self.target_types[k]
        if t == _ie.TargetType.ANTI_AIR:
            F = self._make_F_static()
            Q = np.diag([_ie.EKF_Q_POS_STATIC, _ie.EKF_Q_POS_STATIC,
                        _ie.EKF_Q_VEL_STATIC,  _ie.EKF_Q_VEL_STATIC])
        elif t == _ie.TargetType.TANK:
            F = self._make_F_cv()
            Q = np.diag([_ie.EKF_Q_POS_TANK, _ie.EKF_Q_POS_TANK,
                        _ie.EKF_Q_VEL_TANK,  _ie.EKF_Q_VEL_TANK])
        else:
            F = self._make_F_cv()
            Q = np.diag([_ie.EKF_Q_POS, _ie.EKF_Q_POS,
                        _ie.EKF_Q_VEL, _ie.EKF_Q_VEL])
        return F, Q.astype(np.float64)

    def predict_all(self):
        _ie = self._ie
        for k in range(self.n_targets):
            F, Q = self._get_kf_config(k)
            self.mu[k] = F @ self.mu[k]
            self.P[k]  = F @ self.P[k] @ F.T + Q
            if self.target_types[k] == _ie.TargetType.ANTI_AIR:
                self.mu[k, 2:] = 0.0
                self.P[k, 2:, 2:] = np.eye(2) * 0.01
            self.track_age[k] += 1
            self.P[k] = np.clip(self.P[k], -1e4, 1e4)

    def update(self, k, z_xy, is_camo=False, n_obs=1):
        R = self._compute_R(k, n_obs, is_camo)
        r_noise = np.sqrt(np.diag(R))
        z_noisy = np.asarray(z_xy, dtype=np.float64) + np.random.randn(2) * r_noise

        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        y = z_noisy - H @ self.mu[k]
        S = H @ self.P[k] @ H.T + R
        K = self.P[k] @ H.T @ np.linalg.inv(S)
        self.mu[k] = self.mu[k] + K @ y
        self.P[k]  = (np.eye(4) - K @ H) @ self.P[k]
        self.dwell_count[k] = min(self.dwell_count[k] + 1, self._ie.DWELL_FULL)
        self.track_age[k] = 0


# ── Tracker factory ─────────────────────────────────────────────────────

def make_tracker(tracker_mode, n_targets, target_types, target_positions):
    if tracker_mode == "imm":
        from isaac_env_v13 import TargetBeliefTracker
        return TargetBeliefTracker(n_targets, target_types, target_positions)
    elif tracker_mode == "kf":
        return CVKFTracker(n_targets, target_types, target_positions)
    elif tracker_mode == "raw":
        return RawPositionTracker(n_targets, target_types, target_positions)
    else:
        raise ValueError(f"Unknown tracker mode: {tracker_mode}")


# ══════════════════════════════════════════════════════════════════════
# 공통 RL 에피소드 루프 — dual obs (Graph-Token, eval_v13_ablation.py와 동일)
# ══════════════════════════════════════════════════════════════════════

def _run_rl_episodes_dual_obs(env, agent, ext_tracker, out_path, todo_eps,
                               num_drones, num_targets, n_targets, nt_non_terrain,
                               target_types, n_eval, desc, acc_class):
    from tqdm import tqdm

    ep_bar = tqdm(todo_eps, desc=desc, ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed_torch(ep_idx)
        obs, gobs = env.reset()
        acc  = acc_class(); last_info = {}
        t0   = time.time()

        raw_u    = env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        np_state = np.random.get_state()
        ext_tracker.reset(_target_positions(env))
        np.random.set_state(np_state)

        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]

        for step in range(MAX_STEPS):
            acts, _, _, _, _ = agent.get_action_and_value(
                obs, gobs, deterministic=False)
            (nobs, ngobs), rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step); last_info = info
            obs, gobs = nobs, ngobs

            np_state = np.random.get_state()
            _run_ext_tracker_step(ext_tracker, env)
            np.random.set_state(np_state)

            _collect_ext(ext_tracker, step, n_targets,
                         trP_hist, commit_hist, stale_hist, tracked_hist,
                         T_commit_50, T_trP_below)
            if dones.all(): break

        elapsed  = time.time() - t0
        ep_steps = step + 1
        ep_info  = acc.summarize(last_info, nt_non_terrain)
        ext      = _summarize_ext(ext_tracker, trP_hist, commit_hist,
                                   stale_hist, tracked_hist,
                                   T_commit_50, T_trP_below, target_types, n_targets)

        u_pct       = ep_info.get("uncertainty_pct",  float("nan"))
        drone_prox  = ep_info.get("drone_colls_ep",   0)
        battery     = ep_info.get("mean_battery_ep",  1.0)
        rtb         = ep_info.get("rtb_count_ep",     0)
        depleted    = ep_info.get("depleted_ep",       0)
        active      = ep_info.get("active_at_end",     num_drones)

        _write_row(out_path, _make_csv_line(
            ep_idx, ep_steps, num_drones, num_targets,
            init_unc, u_pct, drone_prox, battery,
            rtb, depleted, active, ext, elapsed))

        ep_bar.set_postfix(ordered_dict={
            "ep":  f"{ep_idx+1:3d}/{n_eval}",
            "U":   f"{u_pct:5.1f}%",
            "trP": f"{ext['ext_mean_trP_final']:.3f}",
            "Cmt": f"{ext['ext_committed_avg']/max(n_targets,1):.2f}",
            "St":  f"{ext['ext_stale_ratio']:.3f}",
        })


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — Graph-Token MAPPO (고정) × tracker variant
# ══════════════════════════════════════════════════════════════════════

def run_graph_token_tracker_scale(scale_key, tracker_mode, out_path, todo_eps,
                                  ckpt_path, n_eval):
    """
    Tier A (metric-only): Graph-Token policy는 그대로, external tracker만 교체.
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from graph_token_mappo_v13 import (TrainConfig, MAPPOAgent,
                                        EpisodeInfoAccumulator,
                                        GraphTokenObsWrapper)

    base_env, _ = _make_env_v13(num_drones, num_targets, k_active)
    env = GraphTokenObsWrapper(base_env)

    target_types   = tuple(int(t.value) if hasattr(t, "value") else int(t)
                           for t in base_env.target_types)
    n_targets      = base_env.num_targets
    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum())

    cfg_agent = TrainConfig()
    agent = MAPPOAgent(cfg_agent)
    agent.load(ckpt_path)
    agent.actor.eval(); agent.critic.eval(); agent.ep_prog = 1.0

    ext_tracker = make_tracker(tracker_mode, n_targets, target_types,
                               _target_positions(base_env))

    _run_rl_episodes_dual_obs(
        env, agent, ext_tracker, out_path, todo_eps,
        num_drones, num_targets, n_targets, nt_non_terrain,
        target_types, n_eval,
        desc=f"  Tracker-{tracker_mode.upper():4s} [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tracker Ablation v13 (metric-only) — raw / kf / imm"
    )
    parser.add_argument("--n_eval",  type=int, default=100)
    parser.add_argument("--restart", action="store_true",
                        help="기존 CSV 삭제 후 처음부터 재실행")
    parser.add_argument("--scales",  nargs="+",
                        choices=["4x20", "8x40", "12x60"],
                        default=["4x20", "8x40", "12x60"])
    parser.add_argument("--methods", nargs="+",
                        choices=TRACKER_METHODS,
                        default=TRACKER_METHODS)
    parser.add_argument("--ckpt_graph_token", type=str, default=None,
                        help="Graph-Token MAPPO 체크포인트 .pt (3 tracker variant 공통)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt_graph_token = args.ckpt_graph_token or find_latest_ckpt(CKPT_DIR, CKPT_PREFIX)
    if not ckpt_graph_token or not os.path.exists(ckpt_graph_token):
        print("  [오류] Graph-Token 체크포인트를 찾을 수 없습니다. "
              "--ckpt_graph_token 으로 직접 지정하세요.")
        return

    METHOD_LABELS = {"raw": "Raw         ", "kf": "KF (CV)     ", "imm": "IMM-KF      "}

    total_combos = len(args.scales) * len(args.methods)
    print(f"\n{'='*72}")
    print(f"  eval_v13_tracker_ablation.py  —  Tracker Ablation (metric-only, Tier A)")
    print(f"  Env      : isaac_env_v13.py  (pure NumPy)")
    print(f"  Policy   : Graph-Token MAPPO (고정)  →  {ckpt_graph_token}")
    print(f"  Scales   : {args.scales}")
    print(f"  Methods  : {args.methods}")
    print(f"  N eval   : {args.n_eval} ep each  |  총 {total_combos * args.n_eval} ep")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*72}")

    combo_idx = 0
    for scale_key in args.scales:
        cfg = SCALE_CONFIGS[scale_key]
        nd, nt = cfg["num_drones"], cfg["num_targets"]

        for tracker_mode in args.methods:
            combo_idx += 1
            csv_name = f"tracker_ablation_{scale_key}_{tracker_mode}.csv"
            out_path = os.path.join(OUT_DIR, csv_name)

            if args.restart and os.path.exists(out_path):
                os.remove(out_path)
                print(f"\n  [Restart] 삭제: {csv_name}")

            done_eps = _get_done_eps(out_path)
            todo_eps = [i for i in range(args.n_eval) if i not in done_eps]

            print(f"\n{'─'*72}")
            print(f"  [{combo_idx}/{total_combos}]  {METHOD_LABELS[tracker_mode]}  |  "
                  f"{scale_key} ({nd} drones / {nt} targets)")
            print(f"  완료: {len(done_eps)} / {args.n_eval}  남은: {len(todo_eps)}")
            print(f"  저장: {out_path}")

            if not todo_eps:
                print("  → 이미 완료. 스킵."); continue

            if not os.path.exists(out_path):
                with open(out_path, "w") as f: f.write(CSV_HEADER)

            run_graph_token_tracker_scale(scale_key, tracker_mode, out_path,
                                          todo_eps, ckpt_graph_token, args.n_eval)

    print(f"\n{'='*72}")
    print(f"  eval_v13_tracker_ablation.py 완료  →  {OUT_DIR}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
