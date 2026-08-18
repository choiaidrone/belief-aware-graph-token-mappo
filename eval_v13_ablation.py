"""
eval_v13_ablation.py — Ablation Study: 4가지 variant × 3 scale
================================================================
[Ablation 구성]
    Variant              method key    사용 코드
    ─────────────────────────────────────────────────────────────────
    Full Proposed        graph_token   graph_token_mappo_v13.py
    w/o Graph Branch     wo_graph      token_ppo_v3.py  (Token-MAPPO)
    w/o Fused Priority   wo_fused      graph_token_wo_fused_v13.py
    w/o IMM-KF/Track     wo_track      graph_token_wo_track_v13.py

[실험 설정]  Fixed Ratio 1:5 (Zero-shot scalability)
    4x20  : 4 drones / 20 targets  ← 학습 환경
    8x40  : 8 drones / 40 targets  ← zero-shot
    12x60 : 12 drones / 60 targets ← zero-shot

[환경]   isaac_env_v13.py  (pure NumPy)

[obs 구조]
    graph_token  : dual obs — (obs484, gobs129)
                   get_action_and_value(obs, gobs, deterministic=False) → 5개 반환
    wo_graph     : single obs — obs484
                   get_action_and_value(obs, deterministic=False)       → 5개 반환
    wo_fused     : dual obs — (obs484[fused→U], gobs129)
                   get_action_and_value(obs, gobs, deterministic=False) → 5개 반환
    wo_track     : dual obs — (obs484[fused→U, track→neutral], gobs129)
                   get_action_and_value(obs, gobs, deterministic=False) → 5개 반환

[CSV 출력]  <repository root>/eval_results/v13/ablation/
    ablation_{scale}_{method}.csv  (총 12개)

[실행법]
    python eval_v13_ablation.py --n_eval 100 ^
        --ckpt_graph_token "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt" ^
        --ckpt_wo_graph    "/path/to/checkpoints/attn_mappo_v12/stage1/mappo_v12_ep03000.pt" ^
        --ckpt_wo_fused    "/path/to/checkpoints/ablation_wo_fused_v13/stage1/wo_fused_ep03000.pt" ^
        --ckpt_wo_track    "/path/to/checkpoints/ablation_wo_track_v13/stage1/wo_track_ep03000.pt"

    # 특정 variant/scale만
    python eval_v13_ablation.py --n_eval 100 --methods graph_token wo_graph --scales 4x20 ^
        --ckpt_graph_token "..." --ckpt_wo_graph "..."

    # 처음부터 다시
    python eval_v13_ablation.py --n_eval 100 --restart --ckpt_graph_token "..." ...
"""

import os, sys, random, time, argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR   = str(REPO_ROOT / "eval_results" / "v13" / "ablation")
MAX_STEPS = 3000

# ── v13: num_anti_air 가변화로 정확한 target 수 사용 가능 ──
SCALE_CONFIGS = {
    "4x20":  {"num_drones": 4,  "num_targets": 20, "k_active": 12},
    "8x40":  {"num_drones": 8,  "num_targets": 40, "k_active": 12},
    "12x60": {"num_drones": 12, "num_targets": 60, "k_active": 12},
}

# 기본 체크포인트 경로 (--ckpt_XXX 미지정 시 자동 탐색; 실제 checkpoint는 repo에
# 포함되지 않으므로, training이 이 repo 안에서 실행되어 같은 이름의 폴더에
# 저장된 경우에만 자동으로 찾는다. 다른 위치의 checkpoint는 --ckpt_XXX 로
# 명시적으로 지정한다.)
CKPT_DIRS = {
    "wo_graph":    str(REPO_ROOT / "attn_mappo_v12" / "stage1"),
    "wo_fused":    str(REPO_ROOT / "ablation_wo_fused_v13" / "stage1"),
    "wo_track":    str(REPO_ROOT / "ablation_wo_track_v13" / "stage1"),
}
CKPT_PREFIXES = {
    "wo_graph":    "mappo_v12_ep",
    "wo_fused":    "wo_fused_ep",
    "wo_track":    "wo_track_ep",
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

        for step in range(MAX_STEPS):
            acts, _, _ = agent.get_action_and_value(obs, deterministic=False)
            nobs, rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step); last_info = info
            obs = nobs
            _run_ext_tracker_step(ext_tracker, env)
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

        for step in range(MAX_STEPS):
            acts, _, _, _, _ = agent.get_action_and_value(
                obs, gobs, deterministic=False)
            (nobs, ngobs), rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step); last_info = info
            obs, gobs = nobs, ngobs
            _run_ext_tracker_step(ext_tracker, env)
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
# 평가 함수 — Full Proposed (Graph-Token MAPPO)
# ══════════════════════════════════════════════════════════════════════

def run_graph_token_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """
    Full Proposed: GraphTokenObsWrapper → (obs484, gobs129)
    agent: MAPPOAgent (graph_token_mappo_v13)
    get_action_and_value(obs, gobs, deterministic=False) → 5개
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from graph_token_mappo_v13 import (TrainConfig, MAPPOAgent,
                                        EpisodeInfoAccumulator,
                                        GraphTokenObsWrapper)
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

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

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                      _target_positions(base_env))

    _run_rl_episodes_dual_obs(
        env, agent, ext_tracker, out_path, todo_eps,
        num_drones, num_targets, n_targets, nt_non_terrain,
        target_types, n_eval,
        desc=f"  Full Proposed [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — w/o Graph Branch (Token-MAPPO, token_ppo_v3)
# ══════════════════════════════════════════════════════════════════════

def run_wo_graph_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """
    w/o Graph Branch: Token-MAPPO (token_ppo_v3).
    obs: single obs484 (base_env.reset() 직접 사용 — wrapper 없음)
    agent: token_ppo_v3.MAPPOAgent
    get_action_and_value(obs, deterministic=False) → (acts, logp, val, aw, cw) 5개
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from token_ppo_v3 import TrainConfig, MAPPOAgent, EpisodeInfoAccumulator
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    env, _ = _make_env_v13(num_drones, num_targets, k_active)

    target_types   = tuple(int(t.value) if hasattr(t, "value") else int(t)
                           for t in env.target_types)
    n_targets      = env.num_targets
    nt_non_terrain = int((~env.ds_map.terrain_mask).sum())

    cfg_agent = TrainConfig()
    agent = MAPPOAgent(cfg_agent)
    agent.load(ckpt_path)
    agent.actor.eval(); agent.critic.eval(); agent.ep_prog = 1.0

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                      _target_positions(env))

    # token_ppo_v3는 single obs, get_action_and_value 반환값 5개
    # → _run_rl_episodes_single_obs의 3개 반환(acts,logp,entropy)과
    #   시그니처가 다르므로 전용 루프 사용
    _run_rl_episodes_token_single(
        env, agent, ext_tracker, out_path, todo_eps,
        num_drones, num_targets, n_targets, nt_non_terrain,
        target_types, n_eval,
        desc=f"  w/o Graph    [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — w/o Fused Priority
# ══════════════════════════════════════════════════════════════════════

def run_wo_fused_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """
    w/o Fused Priority: graph_token_wo_fused_v13.GraphTokenObsWrapper
    fused priority [310:410] → DS U map으로 대체.
    나머지 구조(dual obs, agent)는 Full Proposed와 동일.
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from graph_token_wo_fused_v13 import (TrainConfig, MAPPOAgent,
                                           EpisodeInfoAccumulator,
                                           GraphTokenObsWrapper)
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    base_env, _ = _make_env_v13(num_drones, num_targets, k_active)
    env = GraphTokenObsWrapper(base_env)   # ablation wrapper — fused→U

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
        desc=f"  w/o Fused    [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — w/o IMM-KF / Active Track
# ══════════════════════════════════════════════════════════════════════

def run_wo_track_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """
    w/o IMM-KF / Active Track: graph_token_wo_track_v13.GraphTokenObsWrapper
    fused priority → U map, active track → neutral value.
    환경 내부 tracker는 유지 (reward/metric 공정성).
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from graph_token_wo_track_v13 import (TrainConfig, MAPPOAgent,
                                           EpisodeInfoAccumulator,
                                           GraphTokenObsWrapper)
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    base_env, _ = _make_env_v13(num_drones, num_targets, k_active)
    env = GraphTokenObsWrapper(base_env)   # ablation wrapper — track→neutral

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
        desc=f"  w/o Track    [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — Heuristic (CC-MPSO / SSM)
# ══════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════
# 전용 에피소드 루프 — token_ppo_v3 (single obs, 반환값 5개)
# ══════════════════════════════════════════════════════════════════════

def _run_rl_episodes_token_single(env, agent, ext_tracker, out_path, todo_eps,
                                   num_drones, num_targets, n_targets, nt_non_terrain,
                                   target_types, n_eval, desc, acc_class):
    """
    token_ppo_v3.MAPPOAgent 전용 루프.
    get_action_and_value(obs, deterministic=False) → (acts, logp, val, aw, cw) 5개.
    env는 base_env 그대로 (wrapper 없음) → obs484 단일 반환.
    """
    from tqdm import tqdm

    ep_bar = tqdm(todo_eps, desc=desc, ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed_torch(ep_idx)
        obs  = env.reset()          # shape (num_drones, 484)
        acc  = acc_class(); last_info = {}
        t0   = time.time()

        raw_u    = env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        ext_tracker.reset(_target_positions(env))
        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]

        for step in range(MAX_STEPS):
            acts, _, _, _, _ = agent.get_action_and_value(obs, deterministic=False)
            nobs, rews, dones, info = env.step(acts.cpu().numpy())
            acc.update(info, step); last_info = info
            obs = nobs
            _run_ext_tracker_step(ext_tracker, env)
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
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study v13 — graph_token / wo_graph / wo_fused / wo_track"
    )
    parser.add_argument("--n_eval",  type=int, default=100)
    parser.add_argument("--restart", action="store_true",
                        help="기존 CSV 삭제 후 처음부터 재실행")
    parser.add_argument("--scales",  nargs="+",
                        choices=["4x20", "8x40", "12x60"],
                        default=["4x20", "8x40", "12x60"])
    parser.add_argument("--methods", nargs="+",
                        choices=["wo_graph", "wo_fused", "wo_track"],
                        default=["wo_graph", "wo_fused", "wo_track"])
    parser.add_argument("--ckpt_wo_graph",    type=str, default=None,
                        help="w/o Graph Branch (token_ppo_v3) 체크포인트 .pt")
    parser.add_argument("--ckpt_wo_fused",    type=str, default=None,
                        help="w/o Fused Priority 체크포인트 .pt")
    parser.add_argument("--ckpt_wo_track",    type=str, default=None,
                        help="w/o IMM-KF/Active Track 체크포인트 .pt")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 체크포인트 자동 탐색
    ckpts = {
        "wo_graph":    args.ckpt_wo_graph or find_latest_ckpt(
            CKPT_DIRS["wo_graph"],    CKPT_PREFIXES["wo_graph"]),
        "wo_fused":    args.ckpt_wo_fused or find_latest_ckpt(
            CKPT_DIRS["wo_fused"],    CKPT_PREFIXES["wo_fused"]),
        "wo_track":    args.ckpt_wo_track or find_latest_ckpt(
            CKPT_DIRS["wo_track"],    CKPT_PREFIXES["wo_track"]),
    }

    METHOD_LABELS = {
        "wo_graph":    "w/o Graph    ",
        "wo_fused":    "w/o Fused   ",
        "wo_track":    "w/o Track   ",
    }

    total_combos = len(args.scales) * len(args.methods)
    print(f"\n{'='*72}")
    print(f"  eval_v13_ablation.py  —  Ablation Study")
    print(f"  Env      : isaac_env_v13.py  (pure NumPy)")
    print(f"  Scales   : {args.scales}")
    print(f"  Methods  : {args.methods}")
    print(f"  N eval   : {args.n_eval} ep each  |  총 {total_combos * args.n_eval} ep")
    for m, ck in ckpts.items():
        if m in args.methods:
            print(f"  [{METHOD_LABELS[m]}]: {ck}")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*72}")

    combo_idx = 0
    for scale_key in args.scales:
        cfg = SCALE_CONFIGS[scale_key]
        nd, nt = cfg["num_drones"], cfg["num_targets"]

        for method in args.methods:
            combo_idx += 1
            csv_name = f"ablation_{scale_key}_{method}.csv"
            out_path = os.path.join(OUT_DIR, csv_name)

            if args.restart and os.path.exists(out_path):
                os.remove(out_path)
                print(f"\n  [Restart] 삭제: {csv_name}")

            done_eps = _get_done_eps(out_path)
            todo_eps = [i for i in range(args.n_eval) if i not in done_eps]

            print(f"\n{'─'*72}")
            print(f"  [{combo_idx}/{total_combos}]  {METHOD_LABELS[method]}  |  "
                  f"{scale_key} ({nd} drones / {nt} targets)")
            print(f"  완료: {len(done_eps)} / {args.n_eval}  남은: {len(todo_eps)}")
            print(f"  저장: {out_path}")

            if not todo_eps:
                print("  → 이미 완료. 스킵."); continue

            ck = ckpts[method]
            if not ck or not os.path.exists(ck):
                print(f"  [경고] 체크포인트 없음 — 스킵. --ckpt_{method} 로 지정하세요.")
                continue

            if not os.path.exists(out_path):
                with open(out_path, "w") as f: f.write(CSV_HEADER)

            # ── method별 분기 ──────────────────────────────────────
            if method == "wo_graph":
                run_wo_graph_scale(scale_key, out_path, todo_eps, ck, args.n_eval)
            elif method == "wo_fused":
                run_wo_fused_scale(scale_key, out_path, todo_eps, ck, args.n_eval)
            elif method == "wo_track":
                run_wo_track_scale(scale_key, out_path, todo_eps, ck, args.n_eval)

    print(f"\n{'='*72}")
    print(f"  eval_v13_ablation.py 완료  →  {OUT_DIR}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

