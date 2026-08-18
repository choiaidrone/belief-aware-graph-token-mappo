"""
eval_v13_density_stress.py — Fixed-UAV Target-Density Stress Test
==================================================================
[실험 설정]  Fixed UAV (8대) · Target 수만 증가 → Target-density stress test
    8 drones /  20 targets  ← 드론 대비 저밀도
    8 drones /  40 targets  ← 학습 환경과 동일 비율 (1:5)
    8 drones /  80 targets  ← 고밀도 (1:10)
    8 drones / 120 targets  ← 초고밀도 (1:15)

[목적]
    드론 수를 고정(8)하고 타겟 밀도만 높였을 때 각 방법론의
    Coverage / Tracking 성능 저하 곡선을 비교.
    No retraining — 8x40 학습 체크포인트 그대로 zero-shot 적용.

[환경]   isaac_env_v13.py  (pure NumPy — Isaac/omni 불필요)

[비교 method]
    PPT 이름               코드 method     사용 코드
    ──────────────────────────────────────────────────────────────
    Graph Attention (2025) gat             gat_mappo_v13.py + gat_obs_wrapper_v13.py
    CC-MPSO (2024)         ccmpso          heuristics_v13.py
    SSM (2023)             ssm             heuristics_v13.py
    Random walk            random          eval 내 인라인 구현
    Proposed               graph_token     graph_token_mappo_v13.py

[보고 Metrics]
    uncertainty_pct (↓)        : Final coverage uncertainty
    unc_reduction_pct (↑)      : Uncertainty 감소율
    ext_auc_mean_trP (↓)       : Mean trP AUC (tracking quality)
    committed_ratio (↑)        : Committed target 비율
    stale_ratio_norm (↓)       : Stale target 비율
    drone_prox_events (↓)      : Inter-UAV proximity events
    active_at_end (↑)          : 에피소드 종료 시 active 드론 수

[CSV 출력]
    <repository root>/eval_results/v13_density/
        density_8x20_gat.csv  density_8x20_ccmpso.csv  ...  (총 20개)
    컬럼: drone_colls → drone_prox_events

[실행법]
    # 일반 Python (Isaac 불필요)
    python eval_v13_density_stress.py --n_eval 100 ^
        --ckpt_gat        "/path/to/checkpoints/gat_paper/stage1/gat_paper_ep03000.pt" ^
        --ckpt_graph_token "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt"

    # 특정 method/scale만
    python eval_v13_density_stress.py --n_eval 100 --methods gat ssm --scales 8x20 8x40

    # 처음부터 다시
    python eval_v13_density_stress.py --n_eval 100 --restart ^
        --ckpt_gat        "..." ^
        --ckpt_graph_token "..."
"""

import os, sys, random, time, argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR   = str(REPO_ROOT / "eval_results" / "v13_density")
MAX_STEPS = 3000

# ── Fixed-UAV Density Stress Test: 드론 8 고정, 타겟만 변화 ──────────
# k_active: 학습 환경(8x40, k_active=12)과 동일하게 12로 고정.
#   → GraphTokenObsWrapper / GATObsWrapper의 active 슬롯 차원(K_ACTIVE×6=72)이
#     모델 내부 하드코딩 값(K_ACTIVE=12)과 일치해야 shape 에러가 없음.
#   → 타겟이 20개여도 active 추적 슬롯은 12개 유지 (미달분은 패딩/마스킹).
SCALE_CONFIGS = {
    "8x20":  {"num_drones": 8, "num_targets":  20, "k_active": 12},
    "8x40":  {"num_drones": 8, "num_targets":  40, "k_active": 12},
    "8x80":  {"num_drones": 8, "num_targets":  80, "k_active": 12},
    "8x120": {"num_drones": 8, "num_targets": 120, "k_active": 12},
}

# 기본 체크포인트 경로 (--ckpt_XXX 미지정 시 자동 탐색; 실제 checkpoint는
# repo에 포함되지 않음)
CKPT_DIRS = {
    # 후보 경로를 순서대로 검사 (앞에서부터 첫 매치 사용):
    #   1) 현재 학습 스크립트가 만드는 layout: stage1/seed0/
    #   2) 논문 원본 checkpoint layout(seed 폴더 없음, Zenodo 배포본 등): stage1/
    # seed1~seed4를 뒤섞어 찾지 않도록 두 layout을 recursive 없이 순서대로만 검사한다.
    "gat": [
        str(REPO_ROOT / "gat_paper" / "stage1" / "seed0"),
        str(REPO_ROOT / "gat_paper" / "stage1"),
    ],
    "graph_token": [
        str(REPO_ROOT / "graph_token_mappo_v13" / "stage1" / "seed0"),
        str(REPO_ROOT / "graph_token_mappo_v13" / "stage1"),
    ],
}
CKPT_PREFIXES = {
    "gat":         "gat_paper_ep",
    "graph_token": "graph_token_ep",
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
    """
    import isaac_env_v13
    R_TRACK = isaac_env_v13.R_TRACK_M

    # wrapper면 base env로 내려가기 (GATObsWrapper / GraphTokenObsWrapper)
    base_env = getattr(env, "env", env)

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
            tracker.track_age[k] = min(tracker.track_age[k] + 1, 9999)
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
# 평가 함수 — GAT (Graph Attention, Zhao et al. 2025)
# ══════════════════════════════════════════════════════════════════════

def run_gat_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """
    GAT: GATObsWrapper(base_env) → obs(num_drones, 129)
    agent: GATMAPPOAgent  ·  get_action_and_value(obs, deterministic=True) → acts, logp, entropy
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from gat_mappo_v13        import TrainConfig, GATMAPPOAgent, EpisodeInfoAccumulator
    from gat_obs_wrapper_v13  import GATObsWrapper
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    base_env, _ = _make_env_v13(num_drones, num_targets, k_active)
    env = GATObsWrapper(base_env)

    target_types   = tuple(int(t.value) if hasattr(t, "value") else int(t)
                           for t in base_env.target_types)
    n_targets      = base_env.num_targets
    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum())

    cfg_agent = TrainConfig()
    agent = GATMAPPOAgent(cfg_agent, num_drones)
    agent.load(ckpt_path)
    agent.actor.eval(); agent.critic.eval()

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                      _target_positions(base_env))

    _run_rl_episodes_single_obs(
        env, agent, ext_tracker, out_path, todo_eps,
        num_drones, num_targets, n_targets, nt_non_terrain,
        target_types, n_eval,
        desc=f"  GAT-MAPPO    [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — Graph-Token MAPPO (Proposed)
# ══════════════════════════════════════════════════════════════════════

def run_graph_token_scale(scale_key, out_path, todo_eps, ckpt_path, n_eval):
    """
    Proposed: GraphTokenObsWrapper(base_env) → (obs484, gobs129)
    agent: MAPPOAgent  ·  get_action_and_value(obs, gobs, deterministic=True)
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
        desc=f"  Graph-Token  [{scale_key}] ",
        acc_class=EpisodeInfoAccumulator)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — Heuristic (CC-MPSO / SSM)
# ══════════════════════════════════════════════════════════════════════

def run_heuristic_v13_scale(scale_key, method, out_path, todo_eps, n_eval):
    """
    CC-MPSO / SSM: heuristics_v13 사용.
    proposed 요소(belief_tracker, intensity_map, fused_priority) 미사용.
    DS uncertainty map만 search planning에 허용.
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    from heuristics_v13 import CCMPSOHeuristic, CCMPSOConfig, SSMHeuristic, SSMConfig
    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker
    from tqdm import tqdm

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    env, _   = _make_env_v13(num_drones, num_targets, k_active)
    target_types   = tuple(int(t.value) if hasattr(t, "value") else int(t)
                           for t in env.target_types)
    n_targets      = env.num_targets
    nt_non_terrain = int((~env.ds_map.terrain_mask).sum())

    if method == "ccmpso":
        heuristic = CCMPSOHeuristic(
            env,
            CCMPSOConfig(
                n_particles=10,   # 기본 20 → 10 (eval 속도 최적화)
                max_iters=10,     # 기본 30 → 10
                horizon=5,        # 기본 8  → 5
            )
        )
        label = "CC-MPSO     "
    else:  # ssm
        heuristic = SSMHeuristic(env, SSMConfig())
        label = "SSM         "

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                      _target_positions(env))

    ep_bar = tqdm(todo_eps,
                  desc=f"  {label} [{scale_key}] ",
                  ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed(ep_idx)
        env.reset()
        t0          = time.time()
        drone_prox  = 0; rtb_count = 0; depleted = 0

        raw_u    = env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        heuristic.reset(env)
        ext_tracker.reset(_target_positions(env))
        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]

        # CC-MPSO: 10 step마다 replan (매 step PSO 호출 → 300회로 축소)
        # SSM: 매 step compute_actions
        replan_interval = 10 if method == "ccmpso" else 1
        last_actions = np.zeros(num_drones, dtype=np.int32)

        for step in range(MAX_STEPS):
            if step % replan_interval == 0:
                actions      = heuristic.compute_actions(env)
                last_actions = actions.copy()
            else:
                actions = last_actions
            _, rews, dones, info = env.step(actions)
            drone_prox += len(info.get("drone_collision", []))
            rtb_count  += len(info.get("returning_drones", []))
            depleted   += len(info.get("depleted_drones",  []))
            _run_ext_tracker_step(ext_tracker, env)
            _collect_ext(ext_tracker, step, n_targets,
                         trP_hist, commit_hist, stale_hist, tracked_hist,
                         T_commit_50, T_trP_below)
            if dones.all(): break

        elapsed  = time.time() - t0
        ep_steps = step + 1
        ext      = _summarize_ext(ext_tracker, trP_hist, commit_hist,
                                   stale_hist, tracked_hist,
                                   T_commit_50, T_trP_below, target_types, n_targets)

        u_pct   = float(env.ds_map.total_uncertainty() / max(nt_non_terrain, 1) * 100.0)
        battery = float(np.mean(env.drone_battery))
        active  = int(np.sum(env.drone_active))

        _write_row(out_path, _make_csv_line(
            ep_idx, ep_steps, num_drones, num_targets,
            init_unc, u_pct, drone_prox, battery,
            rtb_count, depleted, active, ext, elapsed))

        ep_bar.set_postfix(ordered_dict={
            "ep":  f"{ep_idx+1:3d}/{n_eval}",
            "U":   f"{u_pct:5.1f}%",
            "trP": f"{ext['ext_mean_trP_final']:.3f}",
            "Cmt": f"{ext['ext_committed_avg']/max(n_targets,1):.2f}",
            "St":  f"{ext['ext_stale_ratio']:.3f}",
        })


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — Random Walk
# ══════════════════════════════════════════════════════════════════════

def run_random_scale(scale_key, out_path, todo_eps, n_eval):
    """
    Random walk: 매 스텝 [0,3] 랜덤 이산 액션 선택.
    특별한 모듈 없이 eval 내에서 인라인 구현.
    """
    cfg = SCALE_CONFIGS[scale_key]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    import isaac_env_v13 as env_module
    from isaac_env_v13 import TargetBeliefTracker
    from tqdm import tqdm

    for k, v in STAGE1_REWARDS.items():
        if hasattr(env_module, k): setattr(env_module, k, v)

    env, _   = _make_env_v13(num_drones, num_targets, k_active)
    target_types   = tuple(int(t.value) if hasattr(t, "value") else int(t)
                           for t in env.target_types)
    n_targets      = env.num_targets
    nt_non_terrain = int((~env.ds_map.terrain_mask).sum())

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                      _target_positions(env))

    ep_bar = tqdm(todo_eps,
                  desc=f"  Random walk  [{scale_key}] ",
                  ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed(ep_idx)
        env.reset()
        t0          = time.time()
        drone_prox  = 0; rtb_count = 0; depleted = 0

        raw_u    = env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        ext_tracker.reset(_target_positions(env))
        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]

        for step in range(MAX_STEPS):
            actions              = np.random.randint(0, 4, size=num_drones)
            _, rews, dones, info = env.step(actions)
            drone_prox += len(info.get("drone_collision", []))
            rtb_count  += len(info.get("returning_drones", []))
            depleted   += len(info.get("depleted_drones",  []))
            _run_ext_tracker_step(ext_tracker, env)
            _collect_ext(ext_tracker, step, n_targets,
                         trP_hist, commit_hist, stale_hist, tracked_hist,
                         T_commit_50, T_trP_below)
            if dones.all(): break

        elapsed  = time.time() - t0
        ep_steps = step + 1
        ext      = _summarize_ext(ext_tracker, trP_hist, commit_hist,
                                   stale_hist, tracked_hist,
                                   T_commit_50, T_trP_below, target_types, n_targets)

        u_pct   = float(env.ds_map.total_uncertainty() / max(nt_non_terrain, 1) * 100.0)
        battery = float(np.mean(env.drone_battery))
        active  = int(np.sum(env.drone_active))

        _write_row(out_path, _make_csv_line(
            ep_idx, ep_steps, num_drones, num_targets,
            init_unc, u_pct, drone_prox, battery,
            rtb_count, depleted, active, ext, elapsed))

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
        description="Fixed-UAV Density Stress Test v13 — GAT / CC-MPSO / SSM / Random / Graph-Token"
    )
    parser.add_argument("--n_eval",  type=int, default=100)
    parser.add_argument("--restart", action="store_true",
                        help="기존 CSV 삭제 후 처음부터 재실행")
    parser.add_argument("--scales",  nargs="+",
                        choices=["8x20", "8x40", "8x80", "8x120"],
                        default=["8x20", "8x40", "8x80", "8x120"])
    parser.add_argument("--methods", nargs="+",
                        choices=["gat", "ccmpso", "ssm", "random", "graph_token"],
                        default=["gat", "ccmpso", "ssm", "random", "graph_token"])
    parser.add_argument("--ckpt_gat", type=str, default=None,
                        help="GAT-MAPPO 체크포인트 .pt 경로")
    parser.add_argument("--ckpt_graph_token", type=str, default=None,
                        help="Graph-Token MAPPO 체크포인트 .pt 경로")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 체크포인트 자동 탐색
    ckpt_gat         = args.ckpt_gat or find_latest_ckpt(
        CKPT_DIRS["gat"],         CKPT_PREFIXES["gat"])
    ckpt_graph_token = args.ckpt_graph_token or find_latest_ckpt(
        CKPT_DIRS["graph_token"], CKPT_PREFIXES["graph_token"])

    total_combos = len(args.scales) * len(args.methods)
    print(f"\n{'='*72}")
    print(f"  eval_v13_density_stress.py  —  Fixed-UAV Target-Density Stress Test")
    print(f"  Env      : isaac_env_v13.py  (pure NumPy, Isaac 불필요)")
    print(f"  UAVs     : 8 고정  |  Targets: {args.scales}")
    print(f"  Methods  : {args.methods}")
    print(f"  N eval   : {args.n_eval} ep each  |  총 {total_combos * args.n_eval} ep")
    print(f"  Safety지표: drone_prox_events (Inter-UAV proximity)")
    if "gat"         in args.methods: print(f"  [GAT]    : {ckpt_gat}")
    if "graph_token" in args.methods: print(f"  [Proposed]: {ckpt_graph_token}")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*72}")

    combo_idx = 0
    for scale_key in args.scales:
        cfg = SCALE_CONFIGS[scale_key]
        nd, nt = cfg["num_drones"], cfg["num_targets"]

        for method in args.methods:
            combo_idx += 1
            csv_name = f"density_{scale_key}_{method}.csv"
            out_path = os.path.join(OUT_DIR, csv_name)

            if args.restart and os.path.exists(out_path):
                os.remove(out_path)
                print(f"\n  [Restart] 삭제: {csv_name}")

            done_eps = _get_done_eps(out_path)
            todo_eps = [i for i in range(args.n_eval) if i not in done_eps]

            method_label = {
                "gat":         "GAT-MAPPO   ",
                "ccmpso":      "CC-MPSO     ",
                "ssm":         "SSM         ",
                "random":      "Random walk ",
                "graph_token": "Graph-Token ",
            }.get(method, method)

            print(f"\n{'─'*72}")
            print(f"  [{combo_idx}/{total_combos}]  {method_label}  |  "
                  f"{scale_key} (8 drones / {cfg['num_targets']} targets)")
            print(f"  완료: {len(done_eps)} / {args.n_eval}  남은: {len(todo_eps)}")
            print(f"  저장: {out_path}")

            if not todo_eps:
                print("  → 이미 완료. 스킵."); continue

            if not os.path.exists(out_path):
                with open(out_path, "w") as f: f.write(CSV_HEADER)

            # ── method별 분기 ──────────────────────────────────────
            if method == "gat":
                if not ckpt_gat or not os.path.exists(ckpt_gat):
                    print("  [경고] GAT 체크포인트 없음 — 스킵. --ckpt_gat 로 지정하세요.")
                    continue
                run_gat_scale(scale_key, out_path, todo_eps, ckpt_gat, args.n_eval)

            elif method == "graph_token":
                if not ckpt_graph_token or not os.path.exists(ckpt_graph_token):
                    print("  [경고] Graph-Token 체크포인트 없음 — 스킵. --ckpt_graph_token 으로 지정하세요.")
                    continue
                run_graph_token_scale(scale_key, out_path, todo_eps,
                                      ckpt_graph_token, args.n_eval)

            elif method in ("ccmpso", "ssm"):
                run_heuristic_v13_scale(scale_key, method, out_path,
                                        todo_eps, args.n_eval)

            elif method == "random":
                run_random_scale(scale_key, out_path, todo_eps, args.n_eval)

    print(f"\n{'='*72}")
    print(f"  eval_v13_density_stress.py 완료  →  {OUT_DIR}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
