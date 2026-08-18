"""
eval_v13_beta_sensitivity.py — β Sensitivity Analysis
=======================================================
[비교 대상]  3개 β 설정 × 3 scale = 9개 CSV
    β=0.0  →  graph_token_mappo_v13_beta0.py   (fused priority: target intensity 미반영)
    β=0.5  →  graph_token_mappo_v13.py         (Full Proposed, 논문 메인 기준)
    β=1.0  →  graph_token_mappo_v13_beta1.py   (target intensity 강하게 반영)

[실험 목적]
    fused priority map의 β가 search-tracking 균형에 미치는 효과 분석.
    fused = U × (1 + β×T) / (1 + β)
      β=0  → search-oriented (U map만 사용)
      β=1  → tracking-oriented (target intensity 강반영)
      β=0.5→ balanced (논문 제안값)

[핵심 지표]
    uncertainty_pct (↓), ext_mean_trP_final (↓),
    committed_ratio (↑), stale_ratio_norm (↓), drone_prox_events (↓)
    + ext_reacquire_count (↑) : stale target 복구 능력

[CSV 출력]  <repository root>/eval_results/v13/beta_sensitivity/
    beta_{beta_tag}_{scale}.csv  (총 9개)
    예: beta_b00_4x20.csv / beta_b05_4x20.csv / beta_b10_4x20.csv

[실행법]
    python eval_v13_beta_sensitivity.py --n_eval 100 ^
        --ckpt_b00 "/path/to/checkpoints/graph_token_beta0_v13/stage1/beta0_ep03000.pt" ^
        --ckpt_b05 "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt" ^
        --ckpt_b10 "/path/to/checkpoints/graph_token_beta1_v13/stage1/beta1_ep03000.pt"

    # 특정 β/scale만
    python eval_v13_beta_sensitivity.py --n_eval 100 --betas b00 b05 --scales 4x20 ^
        --ckpt_b00 "..." --ckpt_b05 "..."

    # 처음부터 다시
    python eval_v13_beta_sensitivity.py --n_eval 100 --restart ^
        --ckpt_b00 "..." --ckpt_b05 "..." --ckpt_b10 "..."
"""

import os, sys, random, time, argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR   = str(REPO_ROOT / "eval_results" / "v13" / "beta_sensitivity")
MAX_STEPS = 3000

# ── v13: num_anti_air 가변화로 정확한 target 수 사용 가능 ──
SCALE_CONFIGS = {
    "4x20":  {"num_drones": 4,  "num_targets": 20, "k_active": 12},
    "8x40":  {"num_drones": 8,  "num_targets": 40, "k_active": 12},
    "12x60": {"num_drones": 12, "num_targets": 60, "k_active": 12},
}

# β별 기본 체크포인트 탐색 경로 (실제 checkpoint는 repo에 포함되지 않음;
# --ckpt_b00/--ckpt_b05/--ckpt_b10 으로 명시 지정 가능)
CKPT_DIRS = {
    "b00": str(REPO_ROOT / "graph_token_beta0_v13" / "stage1"),
    "b05": str(REPO_ROOT / "graph_token_mappo_v13" / "stage1"),
    "b10": str(REPO_ROOT / "graph_token_beta1_v13" / "stage1"),
}
CKPT_PREFIXES = {
    "b00": "beta0_ep",
    "b05": "graph_token_ep",
    "b10": "beta1_ep",
}

STAGE1_REWARDS = {
    "LAMBDA_TRACK":              0.3,
    "LAMBDA_STALE":              0.01,
    "REWARD_UNCERTAINTY_REDUCE": 1.5,
    "PENALTY_TERRAIN":          -0.05,
    "PENALTY_DRONE_COLLISION":  -1.0,
    "ANTIAIR_KILL_PENALTY":      0.0,
    "BATTERY_COST_TRACK":        0.0001,
}

# ── CSV 헤더 ──────────────────────────────────────────────────────────
# drone_colls → drone_prox_events (Safety/Coordination 지표 교수 피드백 반영)
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
# 공통 유틸
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


# ── v13 환경 생성 (Isaac/omni 불필요) ──────────────────────────────
def _make_env_v13(num_drones, num_targets, k_active):
    """
    isaac_env_v13 기반 순수 NumPy 환경 생성.
    v13은 world.reset() / world.step() 불필요.
    DRONE_STARTS 이미 12개 지원 → 별도 패치 없음.
    """
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


# ── 외부 Tracker (공통) ──────────────────────────────────────────────
def _run_ext_tracker_step(tracker, env):
    """
    환경 기준 외부 TargetBeliefTracker 1스텝 업데이트.
    GAT/Graph-Token은 wrapper를 쓰므로 base env로 내려가서
    target_is_camouflaged를 정확히 읽는다.
    중요: predict_all() 먼저 → covariance 증가 반영 후 observation update.
    """
    import isaac_env_v13
    R_TRACK = isaac_env_v13.R_TRACK_M

    # wrapper면 base env로 내려가기 (GATObsWrapper / GraphTokenObsWrapper)
    base_env = getattr(env, "env", env)

    # 환경 step()과 동일한 순서: prediction → observation update
    tracker.predict_all()

    drone_xy = _drone_positions(env)
    tgt_pos  = _target_positions(env)
    camo_arr = getattr(base_env, "target_is_camouflaged",
                       [False] * env.num_targets)

    for k in range(env.num_targets):
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
                   T_commit_50, T_trP_below, target_types, n_targets,
                   reacquire_count=0, GOOD=0.6):
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
    # reacquire_count: 루프 내 step마다 STALE→COMMITTED 전이 누적값 수신
    return {
        "ext_mean_trP_final":        float(ta[-1]) if n else 1.0,
        "ext_auc_mean_trP":          float(np.mean(ta)) if n else 1.0,
        "ext_committed_final":       int(ca[-1]) if n else 0,
        "ext_committed_avg":         float(np.mean(ca)) if n else 0.0,
        "ext_stale_avg":             avg_st,
        "ext_tracked_avg":           avg_trk,
        "ext_stale_ratio":           avg_st / max(avg_trk, 1.0),
        "ext_reacquire_count":       int(reacquire_count),
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
    """
    CSV 한 줄 생성.
    drone_prox: Inter-UAV proximity events (안전거리 위반 횟수)
                — 물리적 충돌이 아닌 coordination 지표
    """
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
# 공통 RL 에피소드 루프 — obs가 단일 tensor인 경우 (GAT용)
# ══════════════════════════════════════════════════════════════════════

def _run_rl_episodes_single_obs(env, agent, ext_tracker, out_path, todo_eps,
                                 num_drones, num_targets, n_targets, nt_non_terrain,
                                 target_types, n_eval, desc, acc_class):
    """GAT: obs 단일 텐서, get_action_and_value(obs, deterministic=True)"""
    from tqdm import tqdm

    ep_bar = tqdm(todo_eps, desc=desc, ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed_torch(ep_idx)
        obs  = env.reset()          # shape (num_drones, GAT_OBS_DIM)
        acc  = acc_class(); last_info = {}
        t0   = time.time()

        raw_u    = env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        ext_tracker.reset(_target_positions(env))
        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]
        reacquire_count = 0
        prev_states = ext_tracker.target_states.copy()

        for step in range(MAX_STEPS):
            acts, _, _ = agent.get_action_and_value(obs, deterministic=False)
            nobs, rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step); last_info = info
            obs = nobs
            _run_ext_tracker_step(ext_tracker, env)
            # STALE → COMMITTED 전이 누적 (episode 전체 reacquire 횟수)
            import isaac_env_v13 as _ie
            reacquire_count += int(np.sum(
                (prev_states == _ie.TargetState.STALE) &
                (ext_tracker.target_states == _ie.TargetState.COMMITTED)
            ))
            prev_states = ext_tracker.target_states.copy()
            _collect_ext(ext_tracker, step, n_targets,
                         trP_hist, commit_hist, stale_hist, tracked_hist,
                         T_commit_50, T_trP_below)
            if dones.all(): break

        elapsed  = time.time() - t0
        ep_steps = step + 1
        ep_info  = acc.summarize(last_info, nt_non_terrain)
        ext      = _summarize_ext(ext_tracker, trP_hist, commit_hist,
                                   stale_hist, tracked_hist,
                                   T_commit_50, T_trP_below, target_types, n_targets,
                                   reacquire_count=reacquire_count)

        u_pct       = ep_info.get("uncertainty_pct",  float("nan"))
        drone_prox  = ep_info.get("drone_colls_ep",   0)   # info key는 동일, CSV 컬럼명만 변경
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


# ── 공통 RL 에피소드 루프 — dual obs (Graph-Token용) ─────────────────

def _run_rl_episodes_dual_obs(env, agent, ext_tracker, out_path, todo_eps,
                               num_drones, num_targets, n_targets, nt_non_terrain,
                               target_types, n_eval, desc, acc_class):
    """
    Graph-Token: GraphTokenObsWrapper 반환값 (obs484, gobs129) 사용.
    get_action_and_value(obs, gobs, deterministic=True)
    """
    from tqdm import tqdm

    ep_bar = tqdm(todo_eps, desc=desc, ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed_torch(ep_idx)
        obs, gobs = env.reset()     # GraphTokenObsWrapper: (obs484, gobs129)
        acc  = acc_class(); last_info = {}
        t0   = time.time()

        raw_u    = env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        ext_tracker.reset(_target_positions(env))
        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]
        reacquire_count = 0
        prev_states = ext_tracker.target_states.copy()

        for step in range(MAX_STEPS):
            acts, _, _, _, _ = agent.get_action_and_value(
                obs, gobs, deterministic=False)
            (nobs, ngobs), rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step); last_info = info
            obs, gobs = nobs, ngobs
            _run_ext_tracker_step(ext_tracker, env)
            # STALE → COMMITTED 전이 누적 (episode 전체 reacquire 횟수)
            import isaac_env_v13 as _ie
            reacquire_count += int(np.sum(
                (prev_states == _ie.TargetState.STALE) &
                (ext_tracker.target_states == _ie.TargetState.COMMITTED)
            ))
            prev_states = ext_tracker.target_states.copy()
            _collect_ext(ext_tracker, step, n_targets,
                         trP_hist, commit_hist, stale_hist, tracked_hist,
                         T_commit_50, T_trP_below)
            if dones.all(): break

        elapsed  = time.time() - t0
        ep_steps = step + 1
        ep_info  = acc.summarize(last_info, nt_non_terrain)
        ext      = _summarize_ext(ext_tracker, trP_hist, commit_hist,
                                   stale_hist, tracked_hist,
                                   T_commit_50, T_trP_below, target_types, n_targets,
                                   reacquire_count=reacquire_count)

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
# 평가 함수 — Full Proposed (Graph-Token MAPPO)
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# β별 평가 함수 공통 내부 로직
# ══════════════════════════════════════════════════════════════════════

def _run_beta_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval,
                    env_module_name, train_module_name, beta_label):
    """
    β 실험 공통 루프.
    각 β마다 env_module과 train_module만 달라지고 나머지는 동일.
    모두 dual obs (obs484, gobs129) 구조 사용.
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    import importlib
    train_mod  = importlib.import_module(train_module_name)
    env_module = importlib.import_module(env_module_name)

    TrainConfig          = train_mod.TrainConfig
    MAPPOAgent           = train_mod.MAPPOAgent
    EpisodeInfoAccumulator = train_mod.EpisodeInfoAccumulator
    GraphTokenObsWrapper = train_mod.GraphTokenObsWrapper
    TargetBeliefTracker  = env_module.TargetBeliefTracker

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    # β별 env로 base_env 생성 — _make_env_v13은 isaac_env_v13 고정이므로 직접 생성
    base_env = env_module.DroneSwarmEnv(
        randomize_terrain=False,
        terrain_seed=42,
        num_drones=num_drones,
        num_targets=num_targets,
        k_active=k_active,
    )
    env = GraphTokenObsWrapper(base_env)

    target_types   = tuple(int(t.value) if hasattr(t, "value") else int(t)
                           for t in base_env.target_types)
    n_targets      = base_env.num_targets
    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum())

    cfg_agent = TrainConfig()
    agent = MAPPOAgent(cfg_agent)
    agent.load(ckpt_path)
    agent.actor.eval(); agent.critic.eval(); agent.ep_prog = 1.0

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                      _target_positions(base_env))

    _run_rl_episodes_dual_obs(
        env, agent, ext_tracker, out_path, todo_eps,
        num_drones, num_targets, n_targets, nt_non_terrain,
        target_types, n_eval,
        desc=f"  β={beta_label} [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


def run_beta00_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """β=0.0 — fused priority: target intensity 미반영 (search-oriented)"""
    _run_beta_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval,
                    env_module_name="isaac_env_v13_beta0",
                    train_module_name="graph_token_mappo_v13_beta0",
                    beta_label="0.0")


def run_beta05_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """β=0.5 — Full Proposed, 논문 메인 기준 (balanced)"""
    _run_beta_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval,
                    env_module_name="isaac_env_v13",
                    train_module_name="graph_token_mappo_v13",
                    beta_label="0.5")


def run_beta10_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """β=1.0 — target intensity 강하게 반영 (tracking-oriented)"""
    _run_beta_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval,
                    env_module_name="isaac_env_v13_beta1",
                    train_module_name="graph_token_mappo_v13_beta1",
                    beta_label="1.0")



# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="β Sensitivity Analysis — β=0.0 / 0.5 / 1.0 × 3 scale"
    )
    parser.add_argument("--n_eval",  type=int, default=100)
    parser.add_argument("--restart", action="store_true",
                        help="기존 CSV 삭제 후 처음부터 재실행")
    parser.add_argument("--scales",  nargs="+",
                        choices=["4x20", "8x40", "12x60"],
                        default=["4x20", "8x40", "12x60"])
    parser.add_argument("--betas",   nargs="+",
                        choices=["b00", "b05", "b10"],
                        default=["b00", "b05", "b10"],
                        help="b00=β0.0  b05=β0.5(proposed)  b10=β1.0")
    parser.add_argument("--ckpt_b00", type=str, default=None,
                        help="β=0.0 체크포인트 .pt")
    parser.add_argument("--ckpt_b05", type=str, default=None,
                        help="β=0.5 체크포인트 .pt (Full Proposed)")
    parser.add_argument("--ckpt_b10", type=str, default=None,
                        help="β=1.0 체크포인트 .pt")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 체크포인트 자동 탐색
    ckpts = {
        "b00": args.ckpt_b00 or find_latest_ckpt(CKPT_DIRS["b00"], CKPT_PREFIXES["b00"]),
        "b05": args.ckpt_b05 or find_latest_ckpt(CKPT_DIRS["b05"], CKPT_PREFIXES["b05"]),
        "b10": args.ckpt_b10 or find_latest_ckpt(CKPT_DIRS["b10"], CKPT_PREFIXES["b10"]),
    }

    BETA_LABELS = {
        "b00": "β=0.0 (search-oriented)",
        "b05": "β=0.5 (balanced, proposed)",
        "b10": "β=1.0 (tracking-oriented)",
    }
    BETA_TAG = {"b00": "b00", "b05": "b05", "b10": "b10"}
    BETA_RUNNERS = {
        "b00": run_beta00_scale,
        "b05": run_beta05_scale,
        "b10": run_beta10_scale,
    }

    total_combos = len(args.betas) * len(args.scales)
    print(f"\n{'='*72}")
    print(f"  eval_v13_beta_sensitivity.py  —  β Sensitivity Analysis")
    print(f"  Env      : β별 isaac_env_v13_betaX.py  (pure NumPy)")
    print(f"  Scales   : {args.scales}")
    print(f"  Betas    : {args.betas}")
    print(f"  N eval   : {args.n_eval} ep each  |  총 {total_combos * args.n_eval} ep")
    for b in args.betas:
        print(f"  [{BETA_LABELS[b]:35s}]: {ckpts[b]}")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*72}")

    combo_idx = 0
    for scale_key in args.scales:
        cfg = SCALE_CONFIGS[scale_key]
        nd, nt = cfg["num_drones"], cfg["num_targets"]

        for beta_key in args.betas:
            combo_idx += 1
            csv_name = f"beta_{BETA_TAG[beta_key]}_{scale_key}.csv"
            out_path = os.path.join(OUT_DIR, csv_name)

            if args.restart and os.path.exists(out_path):
                os.remove(out_path)
                print(f"\n  [Restart] 삭제: {csv_name}")

            done_eps = _get_done_eps(out_path)
            todo_eps = [i for i in range(args.n_eval) if i not in done_eps]

            print(f"\n{'─'*72}")
            print(f"  [{combo_idx}/{total_combos}]  {BETA_LABELS[beta_key]}  |  "
                  f"{scale_key} ({nd} drones / {nt} targets)")
            print(f"  완료: {len(done_eps)} / {args.n_eval}  남은: {len(todo_eps)}")
            print(f"  저장: {out_path}")

            if not todo_eps:
                print("  → 이미 완료. 스킵."); continue

            ck = ckpts[beta_key]
            if not ck or not os.path.exists(ck):
                print(f"  [경고] 체크포인트 없음 — 스킵. --ckpt_{beta_key} 로 지정하세요.")
                continue

            if not os.path.exists(out_path):
                with open(out_path, "w") as f: f.write(CSV_HEADER)

            BETA_RUNNERS[beta_key](scale_key, out_path, todo_eps, ck, args.n_eval)

    print(f"\n{'='*72}")
    print(f"  eval_v13_beta_sensitivity.py 완료  →  {OUT_DIR}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
