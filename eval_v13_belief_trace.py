"""
eval_v13_belief_trace.py — Belief-Evolution Trace (정성적 분석용)
======================================================================================
[목적]
    Graph-Token MAPPO (Proposed) 단일 에피소드를 1회 실행하면서,
    타겟 1개(기본: TANK)의 belief 상태를 매 스텝 CSV로 기록하고,
    DS/fused priority map 스냅샷을 주기적으로 .npz로 저장한다.

    논문 산출물:
        1) IMM mode probability plot   (mode0_prob / mode1_prob vs step)
        2) Belief-evolution 정성적 figure (map snapshot + 추정 위치/공분산 타원)

[베이스]
    eval_v13_scale_all.py 의 구조(SCALE_CONFIGS, STAGE1_REWARDS, _make_env_v13,
    run_graph_token_scale 패턴)를 그대로 사용.
    단, 외부 TargetBeliefTracker 갱신 함수(_run_ext_tracker_step)는
    eval_v13_beta_sensitivity.py 버전(predict_all() 먼저 호출)을 사용한다.
    → predict_all() 을 먼저 호출해야 미관측 구간의 covariance 증가가
      정확히 반영되고, 이후 update()가 그 위에 덧씌워진다.

[실행법]
    python eval_v13_belief_trace.py --scale 4x20 --ep_idx 0 ^
        --ckpt_graph_token "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt"

[출력]
    <repository root>/eval_results/v13_trace/belief_trace_{scale}_ep{idx:03d}.csv
    <repository root>/eval_results/v13_trace/belief_snapshots_{scale}_ep{idx:03d}.npz
"""

import os, sys, random, time, argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR   = str(REPO_ROOT / "eval_results" / "v13_trace")
MAX_STEPS = 3000

SCALE_CONFIGS = {
    "4x20":  {"num_drones": 4,  "num_targets": 20, "k_active": 12},
    "8x40":  {"num_drones": 8,  "num_targets": 40, "k_active": 12},
    "12x60": {"num_drones": 12, "num_targets": 60, "k_active": 12},
}

# 후보 경로를 순서대로 검사: 1) 현재 학습 layout(stage1/seed0/) 2) 논문 원본
# checkpoint layout(seed 폴더 없음, Zenodo 배포본 등, stage1/). seed1~seed4를
# 뒤섞어 찾지 않도록 두 layout을 recursive 없이 순서대로만 검사한다.
CKPT_DIR    = [
    str(REPO_ROOT / "graph_token_mappo_v13" / "stage1" / "seed0"),
    str(REPO_ROOT / "graph_token_mappo_v13" / "stage1"),
]
CKPT_PREFIX = "graph_token_ep"

STAGE1_REWARDS = {
    "LAMBDA_TRACK":              0.3,
    "LAMBDA_STALE":              0.01,
    "REWARD_UNCERTAINTY_REDUCE": 1.5,
    "PENALTY_TERRAIN":          -0.05,
    "PENALTY_DRONE_COLLISION":  -1.0,
    "ANTIAIR_KILL_PENALTY":      0.0,
    "BATTERY_COST_TRACK":        0.0001,
}

# 맵 스냅샷을 저장할 스텝 (기본값) — 필요시 --snapshot_steps 로 override
DEFAULT_SNAPSHOT_STEPS = [50, 200, 500, 1000, 1500, 2000, 2500]

TARGET_TYPE_NAMES = {0: "INFANTRY", 1: "ARTILLERY", 2: "TANK", 3: "ANTI_AIR"}
TARGET_STATE_NAMES = {0: "UNCOMMITTED", 1: "CANDIDATE", 2: "COMMITTED", 3: "STALE"}

CSV_HEADER = (
    "step,target_id,target_type,observed,nearest_uav_dist,"
    "true_x,true_y,est_x,est_y,est_vx,est_vy,"
    "tr_P,tr_P_norm,mode0_prob,mode1_prob,"
    "track_age,track_status,reacquire_event\n"
)


# ══════════════════════════════════════════════════════════════════════
# 공통 유틸 (eval_v13_scale_all.py 동일)
# ══════════════════════════════════════════════════════════════════════

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

def find_latest_ckpt(folders, prefix):
    # folders: 단일 경로(str) 또는 후보 경로 list. 순서대로 첫 번째로
    # prefix*.pt 가 존재하는 디렉터리를 사용한다 (여러 layout을 recursive로
    # 섞어 찾지 않고, 후보 디렉터리 단위로만 순서대로 검사).
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]
    for folder in folders:
        if not os.path.isdir(folder): continue
        pts = [f for f in os.listdir(folder)
               if f.startswith(prefix) and f.endswith(".pt")]
        if not pts: continue
        def _ep(f):
            try: return int(f.replace(prefix, "").replace(".pt", ""))
            except: return -1
        pts.sort(key=_ep)
        path = os.path.join(folder, pts[-1])
        print(f"  [자동 탐색] {path}")
        return path
    return None

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


# ══════════════════════════════════════════════════════════════════════
# 외부 Tracker 1스텝 갱신 — eval_v13_beta_sensitivity.py 버전 (predict_all 먼저)
# ══════════════════════════════════════════════════════════════════════

def _run_ext_tracker_step(tracker, env):
    """
    환경 기준 외부 TargetBeliefTracker 1스텝 업데이트.

    중요: predict_all() 먼저 → covariance 증가 반영 후 observation update.
    (eval_v13_scale_all.py의 구버전은 predict_all()을 호출하지 않아
     미관측 구간의 covariance 증가가 반영되지 않았음 — belief trace에는
     이 버전을 사용해야 정성적으로 의미 있는 covariance 곡선이 나온다.)
    """
    import isaac_env_v13
    R_TRACK = isaac_env_v13.R_TRACK_M

    base_env = getattr(env, "env", env)
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
            # track_age는 predict_all() 내부에서 이미 증가 — 수동 증가 불필요
    tracker.update_target_states()


# ══════════════════════════════════════════════════════════════════════
# 추적 대상 타겟 선택 / 관측 여부 / belief 값 추출
# ══════════════════════════════════════════════════════════════════════

def _class_int_attrs(cls):
    """
    isaac_env_v13.TargetType / TargetState는 enum.Enum이 아니라
    INFANTRY=0 같은 int 클래스 속성만 갖는 일반 class다.
    (.name 같은 Enum API가 없으므로 직접 introspection으로 {값: 이름}을 만든다.)
    """
    return {v: k for k, v in vars(cls).items()
           if not k.startswith("_") and isinstance(v, int)}

def _pick_target_for_trace(target_types, env_module, target_id=None):
    """target_id가 지정되면 그대로 사용, 아니면 TANK 타겟 중 첫 번째를 선택."""
    if target_id is not None:
        return int(target_id)
    TargetType = env_module.TargetType
    tank_ids = [i for i, t in enumerate(target_types) if int(t) == int(TargetType.TANK)]
    if tank_ids:
        return tank_ids[0]
    return 0

def _is_target_observed(env, tid):
    """tid 타겟이 이번 스텝에 R_TRACK_M 이내 드론에게 관측되었는지."""
    import isaac_env_v13
    R_TRACK = isaac_env_v13.R_TRACK_M
    base_env = getattr(env, "env", env)
    drone_xy = _drone_positions(base_env)
    tgt_xy   = np.asarray(base_env.target_positions[tid][:2], dtype=np.float64)
    dists    = np.linalg.norm(drone_xy - tgt_xy, axis=1)
    return int(np.sum(dists < R_TRACK) > 0), float(np.min(dists))

def _safe_get_tracker_trace(tracker, tid):
    """
    TargetBeliefTracker에서 tid 타겟의 belief 값을 꺼낸다.
    isaac_env_v13.TargetBeliefTracker 기준 속성명을 우선 사용하고,
    혹시 다른 변형(beta 버전 등)에서 이름이 다를 경우를 대비해 getattr fallback.
    """
    mu_arr = getattr(tracker, "mu", None)
    if mu_arr is None:
        mu_arr = getattr(tracker, "mus", None)
    P_arr = getattr(tracker, "P", None)
    if P_arr is None:
        P_arr = getattr(tracker, "Ps", None)
    mp_arr = getattr(tracker, "imm_mu_probs", None)
    if mp_arr is None:
        mp_arr = getattr(tracker, "mode_probs", None)

    mu  = np.asarray(mu_arr[tid], dtype=np.float64) if mu_arr is not None else np.zeros(4)
    P   = np.asarray(P_arr[tid],  dtype=np.float64) if P_arr  is not None else np.eye(4)

    # mode probability가 실제로 없으면 [1.0, 0.0]처럼 "가짜 확정값"으로 채우지 않고
    # NaN으로 남겨서 plot/CSV에서 "tracker에 IMM이 없었다"는 사실이 바로 드러나게 한다.
    if mp_arr is not None:
        mp = np.asarray(mp_arr[tid], dtype=np.float64).reshape(-1)
    else:
        mp = np.array([np.nan, np.nan], dtype=np.float64)
    if len(mp) == 1:
        mp = np.array([mp[0], np.nan], dtype=np.float64)

    age = float(tracker.track_age[tid])
    state = int(tracker.target_states[tid])
    return mu, P, mp, age, state

def _make_trace_row(step, tid, ttype_name, observed, near_dist, true_xy,
                    tracker, reacquire_event):
    mu, P, mp, age, state = _safe_get_tracker_trace(tracker, tid)
    trP      = float(np.trace(P))
    trP_norm = tracker.get_tr_P_norm(tid)
    return (
        f"{step},{tid},{ttype_name},{int(observed)},{near_dist:.3f},"
        f"{true_xy[0]:.3f},{true_xy[1]:.3f},"
        f"{mu[0]:.3f},{mu[1]:.3f},{mu[2]:.3f},{mu[3]:.3f},"
        f"{trP:.4f},{trP_norm:.4f},"
        f"{mp[0]:.4f},{mp[1]:.4f},"
        f"{age:.0f},{TARGET_STATE_NAMES.get(state, state)},{int(reacquire_event)}\n"
    )


# ══════════════════════════════════════════════════════════════════════
# 메인 실행 — Graph-Token MAPPO (Proposed) 단일 에피소드
# ══════════════════════════════════════════════════════════════════════

def run_belief_trace(scale_key, ckpt_path, ep_idx, target_id, snapshot_steps,
                     out_csv, out_npz):
    from tqdm import tqdm
    from graph_token_mappo_v13 import (TrainConfig, MAPPOAgent, GraphTokenObsWrapper,
                                        OBS_U_START, OBS_INTENSITY_START, DS_N_RL)
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker, TargetState

    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    base_env, _ = _make_env_v13(num_drones, num_targets, k_active)
    env = GraphTokenObsWrapper(base_env)

    target_types = tuple(int(t.value) if hasattr(t, "value") else int(t)
                         for t in base_env.target_types)

    tid = _pick_target_for_trace(target_types, env_module, target_id)
    try:
        # TargetType이 enum.Enum이 아니라 일반 class라 .name이 없음 →
        # 클래스 속성을 직접 introspect해서 이름을 가져온다 (하드코딩 dict보다 안전).
        ttype_name = _class_int_attrs(env_module.TargetType).get(
            target_types[tid], str(target_types[tid]))
    except Exception:
        ttype_name = TARGET_TYPE_NAMES.get(target_types[tid], str(target_types[tid]))

    cfg_agent = TrainConfig()
    agent = MAPPOAgent(cfg_agent)
    agent.load(ckpt_path)
    agent.actor.eval(); agent.critic.eval(); agent.ep_prog = 1.0

    set_seed_torch(ep_idx)
    obs, gobs = env.reset()

    ext_tracker = TargetBeliefTracker(num_targets, target_types,
                                      _target_positions(base_env))
    ext_tracker.reset(_target_positions(base_env))

    print(f"  추적 대상: target #{tid} ({ttype_name})  |  scale={scale_key}  ep={ep_idx}")

    rows = [CSV_HEADER]
    snap_steps = []; snap_u_maps = []; snap_fused_maps = []
    snap_drone_xy = []; snap_true_xy = []; snap_est_xy = []; snap_est_P = []
    snap_observed = []; snap_track_age = []; snap_track_status = []
    snapshot_set = set(snapshot_steps)

    ep_bar = tqdm(range(MAX_STEPS), desc="  belief trace", ncols=100, unit="step")

    for step in ep_bar:
        acts, _, _, _, _ = agent.get_action_and_value(obs, gobs, deterministic=False)
        (nobs, ngobs), rews, dones, info = env.step(acts.cpu().numpy())

        prev_state_tid = int(ext_tracker.target_states[tid])
        _run_ext_tracker_step(ext_tracker, env)
        reacquire_event = (prev_state_tid == TargetState.STALE and
                           int(ext_tracker.target_states[tid]) == TargetState.COMMITTED)

        observed, near_dist = _is_target_observed(env, tid)
        true_xy = _target_positions(base_env)[tid, :2]

        rows.append(_make_trace_row(step, tid, ttype_name, observed, near_dist,
                                    true_xy, ext_tracker, reacquire_event))

        if step in snapshot_set:
            u_map     = np.asarray(nobs[0, OBS_U_START:OBS_U_START + DS_N_RL*DS_N_RL],
                                   dtype=np.float32).reshape(DS_N_RL, DS_N_RL)
            fused_map = np.asarray(nobs[0, OBS_INTENSITY_START:OBS_INTENSITY_START + DS_N_RL*DS_N_RL],
                                   dtype=np.float32).reshape(DS_N_RL, DS_N_RL)
            mu, P, mp, age, state = _safe_get_tracker_trace(ext_tracker, tid)
            snap_steps.append(step)
            snap_u_maps.append(u_map)
            snap_fused_maps.append(fused_map)
            snap_drone_xy.append(_drone_positions(base_env).copy())
            snap_true_xy.append(true_xy.copy())
            snap_est_xy.append(mu[:2].copy())
            snap_est_P.append(P[:2, :2].copy())
            snap_observed.append(observed)
            snap_track_age.append(age)
            snap_track_status.append(state)

        ep_bar.set_postfix(ordered_dict={
            "trP":  f"{ext_tracker.get_tr_P_norm(tid):.3f}",
            "age":  f"{int(ext_tracker.track_age[tid])}",
            "stat": TARGET_STATE_NAMES.get(int(ext_tracker.target_states[tid]), "?"),
        })

        obs, gobs = nobs, ngobs
        if dones.all():
            break

    n_steps = step + 1

    with open(out_csv, "w") as f:
        f.writelines(rows)
    print(f"  CSV 저장: {out_csv}  ({n_steps} steps)")

    if snap_steps:
        np.savez(
            out_npz,
            steps=np.array(snap_steps, dtype=np.int32),
            u_maps=np.stack(snap_u_maps),
            fused_maps=np.stack(snap_fused_maps),
            drone_xy=np.stack(snap_drone_xy),
            true_xy=np.stack(snap_true_xy),
            est_xy=np.stack(snap_est_xy),
            est_P=np.stack(snap_est_P),
            observed=np.array(snap_observed, dtype=bool),
            track_age=np.array(snap_track_age, dtype=np.float32),
            track_status=np.array(snap_track_status, dtype=np.int32),
            target_id=tid,
            target_type=ttype_name,
            scale=scale_key,
            ep_idx=ep_idx,
        )
        print(f"  Snapshot 저장: {out_npz}  ({len(snap_steps)} snapshots)")
    else:
        print("  [경고] snapshot_steps가 에피소드 길이보다 커서 저장된 스냅샷이 없습니다.")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Belief-Evolution Trace — Graph-Token MAPPO (Proposed) 단일 에피소드"
    )
    parser.add_argument("--scale", type=str, default="4x20",
                        choices=list(SCALE_CONFIGS.keys()))
    parser.add_argument("--ep_idx", type=int, default=0,
                        help="에피소드 시드 (set_seed_torch에 사용)")
    parser.add_argument("--target_id", type=int, default=None,
                        help="추적할 target index. 미지정 시 TANK 타겟 자동 선택")
    parser.add_argument("--ckpt_graph_token", type=str, default=None,
                        help="Graph-Token MAPPO 체크포인트 .pt 경로")
    parser.add_argument("--snapshot_steps", type=int, nargs="+", default=None,
                        help=f"맵 스냅샷 저장 스텝 목록 (기본: {DEFAULT_SNAPSHOT_STEPS})")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt_path = args.ckpt_graph_token or find_latest_ckpt(CKPT_DIR, CKPT_PREFIX)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("  [오류] Graph-Token 체크포인트를 찾을 수 없습니다. "
              "--ckpt_graph_token 으로 직접 지정하세요.")
        return

    snapshot_steps = args.snapshot_steps or DEFAULT_SNAPSHOT_STEPS

    out_csv = os.path.join(OUT_DIR, f"belief_trace_{args.scale}_ep{args.ep_idx:03d}.csv")
    out_npz = os.path.join(OUT_DIR, f"belief_snapshots_{args.scale}_ep{args.ep_idx:03d}.npz")

    print(f"\n{'='*72}")
    print("  eval_v13_belief_trace.py — Belief-Evolution Trace (Proposed)")
    print(f"  Scale         : {args.scale}")
    print(f"  Episode (seed): {args.ep_idx}")
    print(f"  Target id     : {'auto(TANK)' if args.target_id is None else args.target_id}")
    print(f"  Ckpt          : {ckpt_path}")
    print(f"  Snapshot steps: {snapshot_steps}")
    print(f"  Output        : {out_csv}")
    print(f"                  {out_npz}")
    print(f"{'='*72}\n")

    run_belief_trace(args.scale, ckpt_path, args.ep_idx, args.target_id,
                     snapshot_steps, out_csv, out_npz)

    print(f"\n{'='*72}")
    print("  완료.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
