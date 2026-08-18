"""
eval_v13_belief_feature_masking.py — Evaluation-Time Belief-Feature Masking Analysis
======================================================================================
[목적]
    재학습 없음. 학습이 끝난 Full Proposed (Graph-Token MAPPO) checkpoint는 고정하고,
    policy가 받는 obs484 중 belief-related feature(trP_norm / age_norm / fused priority)
    일부만 evaluation 시점에 가려서(mask) 성능이 떨어지는지를 본다.
        -> "Policy가 belief information을 실제로 쓰는가?"
    논문 리뷰 5번에 대응하는 실험.

    eval_v13_ablation.py의 wo_fused / wo_track과 이름은 비슷하지만 성격이 다르다:
    그쪽은 학습 시점부터 다른 obs로 새로 학습한 "별도 checkpoint"를 평가하는 것이고,
    이 파일은 "동일한 학습된 가중치"에 evaluation 시점에만 입력을 가린다
    (= policy-aware ablation이 아니라 evaluation-time feature masking).

[variant]  (총 6개)
    full          : 마스킹 없음 (baseline)
    wo_trp        : active-track token에서 trP_norm(인덱스 4)만 neutral(0.5)
    wo_age        : active-track token에서 age_norm(인덱스 5)만 neutral(0.5)
    wo_trp_age    : trP_norm + age_norm 모두 neutral
    wo_fused      : fused priority map[310:410] -> raw DS uncertainty map[200:300]
    wo_all_belief : wo_trp_age + wo_fused 동시 적용

    active-track token 구조 (6dim x K_ACTIVE=12, isaac_env_v13.active_track_obs_vector):
        [dx, dy, vx, vy, trP_norm, age_norm]
    주의: token 안에는 "track status"가 직접 들어있지 않다. status는
    get_active_tracks()에서 priority score(어떤 타겟을 active-track 12개 슬롯에
    넣을지)를 정할 때만 쓰이고, token 값 자체에는 포함되지 않는다. 그래서
    "w/o track status" 라고 쓰면 부정확하다. wo_fused는 status-dependent weighting만
    제거하는 게 아니라, target-belief-weighted fused priority map 전체(intensity +
    status-dependent weighting 모두 포함)를 raw DS uncertainty로 대체하는 것이므로
    정확히는 다음과 같이 표현해야 한다:
        "w/o fused priority removes the target-belief-weighted priority map,
         including the effect of status-dependent priority weighting."

[obs484 인덱스]
    isaac_env_v13.py 기준: OBS_DS_DIM=300, OBS_DEPTH_DIM=8, OBS_SELF_DIM=2,
    OBS_INTENSITY_DIM=100, OBS_BELIEF_START=410, OBS_ACTIVE_DIM=72, OBS_BATTERY_DIM=2.

        [  0:100] DS belief map E (existence), 10x10 pooled, flatten
        [100:200] DS belief map F (false),      10x10 pooled, flatten
        [200:300] DS belief map U (uncertainty), 10x10 pooled, flatten  <- wo_fused 원본
        [300:308] depth (8방향 ray-cast)
        [308:310] self position (x, y)
        [310:410] fused priority map (10x10=100, get_fused_priority(U_pool) 출력)
        [410:482] active tracks (K_ACTIVE=12 x 6)
        [482:484] battery [battery_norm, dist_to_base_norm]

    DS map(E,F,U)은 cell마다 3채널이 interleave된 (10,10,3) 구조가 아니라,
    DSBeliefMap.belief_vector() = concat(E_pool.flatten(), F_pool.flatten(), U_pool.flatten())
    형태로 100dim 블록 3개가 그대로 이어붙은 구조다. 따라서 U_pool은
    obs[:, 200:300]을 reshape 없이 그대로 슬라이스하면 된다 — 이게 정확히
    _get_obs_single()에서 get_fused_priority(U_pool)에 넘겨준 그 배열과 동일하다.

[gobs]
    Graph-Token의 global obs(129dim)는 이번 실험에서 건드리지 않는다
    (obs484 안의 belief feature만 다룬다).

[external tracker / RNG]
    metric 계산용 ext_tracker(TargetBeliefTracker)는 env의 internal belief_tracker와
    별개의 인스턴스다. ext_tracker.reset()/update()가 쓰는 np.random 호출이 env 자체의
    random stream(타겟 모션 등)을 건드리지 않도록 np.random.get_state()/set_state()로
    감싼다 (eval_v13_tracker_ablation.py와 동일한 보호 패턴).

[CSV 출력]  <repository root>/eval_results/v13/belief_feature_masking/
    belief_feature_masking_{scale}_{variant}.csv  (6 variant x 3 scale = 18개)

[기준 파일]
    eval_v13_ablation.py의 구조(SCALE_CONFIGS, STAGE1_REWARDS, CSV_HEADER, resume 로직,
    _run_ext_tracker_step의 predict_all() 우선 순서, _run_rl_episodes_dual_obs)를
    그대로 가져왔다.

[실행법]
    # smoke test
    python eval_v13_belief_feature_masking.py --n_eval 3 --scales 4x20 ^
        --methods full wo_trp wo_age wo_trp_age wo_fused wo_all_belief ^
        --ckpt_graph_token "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt" --restart

    # full run
    python eval_v13_belief_feature_masking.py --n_eval 100 --scales 4x20 8x40 12x60 ^
        --ckpt_graph_token "/path/to/checkpoints/graph_token_mappo_v13/stage1/graph_token_ep03000.pt" --restart
"""

import os, sys, random, time, argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR   = str(REPO_ROOT / "eval_results" / "v13" / "belief_feature_masking")
MAX_STEPS = 3000

SCALE_CONFIGS = {
    "4x20":  {"num_drones": 4,  "num_targets": 20, "k_active": 12},
    "8x40":  {"num_drones": 8,  "num_targets": 40, "k_active": 12},
    "12x60": {"num_drones": 12, "num_targets": 60, "k_active": 12},
}

# Full Proposed checkpoint 고정 — variant끼리 비교 대상이 아니므로 ckpt 1개만 사용
# (실제 checkpoint는 repo에 포함되지 않음; --ckpt_graph_token 으로 명시 지정 가능)
# 후보 경로를 순서대로 검사: 1) 현재 학습 layout(stage1/seed0/) 2) 논문 원본
# checkpoint layout(seed 폴더 없음, Zenodo 배포본 등, stage1/). seed1~seed4를
# 뒤섞어 찾지 않도록 두 layout을 recursive 없이 순서대로만 검사한다.
CKPT_DIR    = [
    str(REPO_ROOT / "graph_token_mappo_v13" / "stage1" / "seed0"),
    str(REPO_ROOT / "graph_token_mappo_v13" / "stage1"),
]
CKPT_PREFIX = "graph_token_ep"

MASK_METHODS = ["full", "wo_trp", "wo_age", "wo_trp_age", "wo_fused", "wo_all_belief"]

STAGE1_REWARDS = {
    "LAMBDA_TRACK":              0.3,
    "LAMBDA_STALE":              0.01,
    "REWARD_UNCERTAINTY_REDUCE": 1.5,
    "PENALTY_TERRAIN":          -0.05,
    "PENALTY_DRONE_COLLISION":  -1.0,
    "ANTIAIR_KILL_PENALTY":      0.0,
    "BATTERY_COST_TRACK":        0.0001,
}

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


# ── 외부 Tracker 1스텝 갱신 (eval_v13_ablation.py 동일 + base_env 안전 처리) ──
def _run_ext_tracker_step(tracker, env):
    """
    predict_all() → observation update → update_target_states() 순서.
    GraphTokenObsWrapper는 속성을 base_env로 forwarding하지만, wrapper 종류에
    상관없이 안전하도록 base_env를 명시적으로 사용한다.
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
# Belief-feature masking (evaluation-time only, 재학습 없음)
# ══════════════════════════════════════════════════════════════════════

def mask_belief_features(obs, variant, k_active=12):
    """
    Evaluation-time feature masking for Graph-Token MAPPO observation (obs484).

    obs shape: (num_drones, 484)
        [  0:100] DS belief map E, 10x10 pooled flatten
        [100:200] DS belief map F, 10x10 pooled flatten
        [200:300] DS belief map U, 10x10 pooled flatten   (raw uncertainty)
        [300:308] depth
        [308:310] self position
        [310:410] fused priority map (10x10=100, belief-weighted)
        [410:482] active tracks (K_ACTIVE x 6): [dx, dy, vx, vy, trP_norm, age_norm]
        [482:484] battery

    variant:
        full          : no masking
        wo_trp        : neutralize trP_norm (covariance trace) in active-track tokens
        wo_age        : neutralize age_norm (track age) in active-track tokens
        wo_trp_age    : neutralize both trP_norm and age_norm
        wo_fused      : replace fused priority map with raw DS uncertainty map (U_pool)
        wo_all_belief : wo_trp_age + wo_fused combined
    """
    if variant == "full":
        return obs

    obs_m = np.array(obs, copy=True)

    # ------------------------------------------------------------
    # 1) Replace fused priority map with raw DS uncertainty map.
    #    obs[:, 200:300] IS U_pool already (flat 100dim block, not interleaved
    #    per-cell with E/F) — it's the exact array get_fused_priority(U_pool)
    #    was given to build obs[:, 310:410]. No reshape needed, just copy.
    # ------------------------------------------------------------
    if variant in ("wo_fused", "wo_all_belief"):
        obs_m[:, 310:410] = obs_m[:, 200:300]

    # ------------------------------------------------------------
    # 2) Mask active-track belief features
    # ------------------------------------------------------------
    if variant in ("wo_trp", "wo_age", "wo_trp_age", "wo_all_belief"):
        tracks = obs_m[:, 410:482].reshape(obs_m.shape[0], k_active, 6)

        # Neutral value: 0.5, because trP_norm/age_norm are normalized to [0,1].
        # 0.0 would read as "perfectly confident / brand-new track" — a strong
        # misleading signal, not an absence of information.
        if variant in ("wo_trp", "wo_trp_age", "wo_all_belief"):
            tracks[:, :, 4] = 0.5

        if variant in ("wo_age", "wo_trp_age", "wo_all_belief"):
            tracks[:, :, 5] = 0.5

        obs_m[:, 410:482] = tracks.reshape(obs_m.shape[0], k_active * 6)

    return obs_m


# ── RL 에피소드 루프 — dual obs (Graph-Token) + belief-feature masking ──

def _run_rl_episodes_masked_dual_obs(env, agent, ext_tracker, out_path, todo_eps,
                                      num_drones, num_targets, n_targets, nt_non_terrain,
                                      target_types, n_eval, desc, acc_class,
                                      variant, k_active):
    """
    eval_v13_ablation.py의 _run_rl_episodes_dual_obs와 동일하지만, policy가 action을
    고르기 직전에 obs484에 mask_belief_features()를 적용한다. gobs(129)는 그대로 둔다.

    ext_tracker(metric 계산용)는 env의 random stream(타겟 모션 등)을 건드리면 안 되므로
    reset()/_run_ext_tracker_step() 호출을 np.random 상태 저장/복원으로 감싼다.
    """
    from tqdm import tqdm

    ep_bar = tqdm(todo_eps, desc=desc, ncols=140,
                  bar_format="{desc}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in ep_bar:
        set_seed_torch(ep_idx)
        obs, gobs = env.reset()     # GraphTokenObsWrapper: (obs484, gobs129)
        obs  = mask_belief_features(obs, variant, k_active=k_active)
        acc  = acc_class(); last_info = {}
        t0   = time.time()

        base_env = getattr(env, "env", env)

        raw_u    = base_env.ds_map.total_uncertainty()
        init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

        np_state = np.random.get_state()
        ext_tracker.reset(_target_positions(base_env))
        np.random.set_state(np_state)

        trP_hist=[]; commit_hist=[]; stale_hist=[]; tracked_hist=[]
        T_commit_50=[None]; T_trP_below=[None]

        for step in range(MAX_STEPS):
            acts, _, _, _, _ = agent.get_action_and_value(
                obs, gobs, deterministic=False)
            (nobs, ngobs), rews, dones, info = env.step(acts.cpu().numpy())
            nobs = mask_belief_features(nobs, variant, k_active=k_active)
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
# 평가 함수 — Full Proposed 체크포인트 고정 + belief-feature masking
# ══════════════════════════════════════════════════════════════════════

def run_graph_token_mask_scale(scale_key, variant, out_path, todo_eps, ckpt_path, n_eval):
    """
    Full Proposed: GraphTokenObsWrapper → (obs484, gobs129).
    checkpoint는 모든 variant가 동일하게 공유 — masking은 evaluation 시점 입력 조작일 뿐,
    policy 가중치는 전혀 바뀌지 않는다.
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

    _run_rl_episodes_masked_dual_obs(
        env, agent, ext_tracker, out_path, todo_eps,
        num_drones, num_targets, n_targets, nt_non_terrain,
        target_types, n_eval,
        desc=f"  Mask-{variant:13s}[{scale_key}] ",
        acc_class=EpisodeInfoAccumulator,
        variant=variant, k_active=k_active)


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation-time belief-feature masking — Graph-Token MAPPO (체크포인트 고정)"
    )
    parser.add_argument("--n_eval",  type=int, default=100)
    parser.add_argument("--restart", action="store_true",
                        help="기존 CSV 삭제 후 처음부터 재실행")
    parser.add_argument("--scales",  nargs="+",
                        choices=["4x20", "8x40", "12x60"],
                        default=["4x20", "8x40", "12x60"])
    parser.add_argument("--methods", nargs="+",
                        choices=MASK_METHODS,
                        default=MASK_METHODS)
    parser.add_argument("--ckpt_graph_token", type=str, default=None,
                        help="Graph-Token MAPPO 체크포인트 .pt (고정, 모든 variant 공유)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = args.ckpt_graph_token or find_latest_ckpt(CKPT_DIR, CKPT_PREFIX)
    if not ckpt or not os.path.exists(ckpt):
        print("  [오류] Graph-Token 체크포인트를 찾을 수 없습니다. --ckpt_graph_token 으로 지정하세요.")
        return

    total_combos = len(args.scales) * len(args.methods)
    print(f"\n{'='*72}")
    print(f"  eval_v13_belief_feature_masking.py  —  Belief-Feature Masking Analysis")
    print(f"  Env      : isaac_env_v13.py  (pure NumPy)")
    print(f"  Ckpt     : {ckpt}  (고정 — 모든 variant 공유, 재학습 없음)")
    print(f"  Scales   : {args.scales}")
    print(f"  Methods  : {args.methods}")
    print(f"  N eval   : {args.n_eval} ep each  |  총 {total_combos * args.n_eval} ep")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*72}")

    combo_idx = 0
    for scale_key in args.scales:
        cfg = SCALE_CONFIGS[scale_key]
        nd, nt = cfg["num_drones"], cfg["num_targets"]

        for method in args.methods:
            combo_idx += 1
            csv_name = f"belief_feature_masking_{scale_key}_{method}.csv"
            out_path = os.path.join(OUT_DIR, csv_name)

            if args.restart and os.path.exists(out_path):
                os.remove(out_path)
                print(f"\n  [Restart] 삭제: {csv_name}")

            done_eps = _get_done_eps(out_path)
            todo_eps = [i for i in range(args.n_eval) if i not in done_eps]

            print(f"\n{'─'*72}")
            print(f"  [{combo_idx}/{total_combos}]  {method:13s}  |  "
                  f"{scale_key} ({nd} drones / {nt} targets)")
            print(f"  완료: {len(done_eps)} / {args.n_eval}  남은: {len(todo_eps)}")
            print(f"  저장: {out_path}")

            if not todo_eps:
                print("  → 이미 완료. 스킵."); continue

            if not os.path.exists(out_path):
                with open(out_path, "w") as f: f.write(CSV_HEADER)

            run_graph_token_mask_scale(scale_key, method, out_path, todo_eps,
                                        ckpt, args.n_eval)

    print(f"\n{'='*72}")
    print(f"  eval_v13_belief_feature_masking.py 완료  →  {OUT_DIR}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
