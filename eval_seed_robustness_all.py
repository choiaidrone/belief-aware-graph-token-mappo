"""
eval_seed_robustness_all.py
============================
Multi-seed robustness evaluation: Graph-Token MAPPO vs GAT-MAPPO.

목적
  각 method(Graph-Token / GAT-MAPPO)를 5개 independently-trained training seed
  checkpoint에 대해, 3개 scale(4x20 / 8x40 / 12x60)에서 각 100 stochastic
  evaluation episode로 평가한다.

  헤드라인 통계는 "500 episode pooling"이 아니라 "training seed"를 통계 단위로
  삼는다:
      1. 각 checkpoint(=method x train_seed x scale)마다 100 episode 평가
      2. 각 checkpoint 안에서 100 episode 평균을 낸다 (per_seed_summary.csv)
      3. 5개 train_seed의 평균을 다시 평균 -> mean +/- std across 5 seeds
         (seed_level_summary.csv)
      4. Graph-Token vs GAT-MAPPO는 5개 seed-level 평균값을 짝지어
         Wilcoxon signed-rank test로 비교한다 (seed_level_tests.csv)

  100-episode raw 결과는 per_episode/*.csv, per_episode_metrics.csv로 별도
  저장한다 (투명성/재분석/진단용. 논문 headline에는 쓰지 않는다).

재사용
  env 생성, external tracker(trP/committed/stale) step/summarize 로직은
  eval_v13_scale_all.py의 기존 구현을 그대로 import해서 쓴다. 지표 정의가
  기존 scale 실험(zero-shot 4x20/8x40/12x60) 결과와 어긋나지 않게 하기 위함.

평가 프로토콜
  - stochastic action sampling (deterministic=False) — 기존 eval_v13_scale_all.py와
    동일. 두 method 모두 같은 프로토콜.
  - eval_seed = 10000 + scale_id*1000 + episode_idx
    -> 같은 (scale, episode_idx)면 method/train_seed에 상관없이 동일 eval_seed.
       env target 배치 / 노이즈 / 액션 샘플링 모두 이 seed로 재현 가능.

Resume
  per_episode/{method}_seed{seed}_{scale}.csv 에 이미 있는 eval_episode는
  건너뛴다. 오래 걸리는 작업이라 중간에 끊겨도 이어서 할 수 있음.

실행 (기본: stage1, 1500ep checkpoint)
    cd /path/to/repository
    conda activate drone_env_310
    python eval_seed_robustness_all.py ^
      --root . ^
      --run_root ./seed_robustness_runs ^
      --out_dir ./seed_robustness_eval ^
      --episodes 100 ^
      --checkpoint_ep 1500 ^
      --device cuda

(인자를 생략하면 위 값들은 모두 repository root 기준 기본값으로 자동 설정된다.)

3000ep checkpoint 평가 시 (--out_dir만 바꾸면 결과가 안 섞임):
    python eval_seed_robustness_all.py ^
      --out_dir ./seed_robustness_eval_ep3000 ^
      --episodes 100 --checkpoint_ep 3000 --device cuda

집계만 다시 하고 싶을 때 (per-episode CSV는 이미 있고 seed_level 표만 재생성):
    python eval_seed_robustness_all.py --aggregate_only --out_dir ./seed_robustness_eval
"""
import os
import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

# 기존 scale 실험 코드의 env/tracker/metric 헬퍼를 그대로 재사용한다.
# (지표 정의가 기존 main scale 결과와 어긋나지 않도록)
import eval_v13_scale_all as base_eval

MAX_STEPS      = base_eval.MAX_STEPS            # 3000
SCALE_CONFIGS  = base_eval.SCALE_CONFIGS        # {"4x20":{...}, "8x40":{...}, "12x60":{...}}
SCALE_ORDER    = ["4x20", "8x40", "12x60"]       # scale_id 매핑 순서 (eval_seed 계산용)

TRAIN_SEEDS_DEFAULT = [0, 1, 2, 3, 4]

# ── method별 checkpoint 위치 / 코드 매핑 ────────────────────────────────
# checkpoint 경로는 {seed}, {ep} 를 조합해서 만든다. RUN_ROOT는 CLI --run_root로 지정.
METHOD_SPECS = {
    "graph_token": {
        "ckpt_dirname":  "graph_token",
        "ckpt_prefix":   "graph_token_ep",
    },
    "gat_mappo": {
        "ckpt_dirname":  "gat_mappo",
        "ckpt_prefix":   "gat_paper_ep",
    },
}

# 방향이 알려진 headline 지표만 better_method를 판정한다.
# True  = Graph-Token 값이 클수록 좋음
# False = Graph-Token 값이 작을수록 좋음
METRIC_DIRECTION_HIGHER_BETTER = {
    "committed_ratio":        True,
    "stale_ratio":            False,
    "final_committed_ratio":  True,
    "final_stale_ratio":      False,
    "avg_committed_ratio":    True,
    "avg_stale_ratio":        False,
    "final_uncertainty":      False,
    "auc_mean_trP":           False,
    "prox_per_1000":          False,
}

# committed_ratio / stale_ratio 는 이름이 헷갈리기 쉬워서 final/avg 버전을
# 둘 다 명시적으로 저장한다. committed_ratio == final_committed_ratio,
# stale_ratio == final_stale_ratio (episode 마지막 시점 기준).
# 논문에서 AUC/평균 성격의 값을 쓰고 싶으면 avg_committed_ratio /
# avg_stale_ratio (episode 평균 기준)를 쓴다.
PER_EPISODE_METRIC_COLS = [
    "uncertainty_reduction", "final_uncertainty", "final_mean_trP", "auc_mean_trP",
    "committed_ratio", "stale_ratio",
    "final_committed_ratio", "final_stale_ratio",
    "avg_committed_ratio", "avg_stale_ratio",
    "time_under_good_track", "prox_per_1000",
    "episode_return", "episode_length",
    "collision_count", "terrain_hit_count", "reacquire_count",
    "final_committed_count", "final_stale_count",
]

CSV_COLUMNS = (
    ["method", "train_seed", "scale", "eval_episode", "eval_seed", "checkpoint_path"]
    + PER_EPISODE_METRIC_COLS
)


# ══════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════

def make_eval_seed(scale_name: str, episode_idx: int) -> int:
    scale_id = SCALE_ORDER.index(scale_name)
    return 10000 + scale_id * 1000 + episode_idx


def _get_done_episodes(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(int(x) for x in df["eval_episode"].tolist())
    except Exception:
        return set()


def _write_row(csv_path: Path, row: dict, write_header: bool):
    """csv.DictWriter로 안전하게 append (comma/quote 포함 값도 안전)."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


# ══════════════════════════════════════════════════════════════════════
# 환경 / 정책 로딩
# ══════════════════════════════════════════════════════════════════════

def build_env(method: str, scale_name: str):
    """
    scale 설정(4x20/8x40/12x60, k_active=12 고정)으로 base env를 만들고
    method에 맞는 obs wrapper로 감싼다.
    """
    cfg = SCALE_CONFIGS[scale_name]
    num_drones, num_targets, k_active = cfg["num_drones"], cfg["num_targets"], cfg["k_active"]

    base_env, _ = base_eval._make_env_v13(num_drones, num_targets, k_active)

    if method == "gat_mappo":
        from gat_obs_wrapper_v13 import GATObsWrapper
        env = GATObsWrapper(base_env)
    elif method == "graph_token":
        from graph_token_mappo_v13 import GraphTokenObsWrapper
        env = GraphTokenObsWrapper(base_env)
    else:
        raise ValueError(f"unknown method: {method}")

    return base_env, env, num_drones, num_targets


def load_policy(method: str, checkpoint_path: str, num_drones: int, device: str):
    """
    Graph-Token 또는 GAT-MAPPO 정책을 로드한다.
    4x20/8x40/12x60 전부 같은 checkpoint로 zero-shot 평가 가능
    (기존 eval_v13_scale_all.py와 동일 방식).
    """
    if method == "gat_mappo":
        from gat_mappo_v13 import TrainConfig, GATMAPPOAgent
        cfg = TrainConfig()
        cfg.device = device
        agent = GATMAPPOAgent(cfg, num_drones)
        agent.load(checkpoint_path)
        agent.actor.eval(); agent.critic.eval()
        return agent

    elif method == "graph_token":
        from graph_token_mappo_v13 import TrainConfig, MAPPOAgent
        cfg = TrainConfig()
        cfg.device = device
        agent = MAPPOAgent(cfg)
        agent.load(checkpoint_path)
        agent.actor.eval(); agent.critic.eval()
        agent.ep_prog = 1.0
        return agent

    raise ValueError(f"unknown method: {method}")


# ══════════════════════════════════════════════════════════════════════
# 1 episode 실행 — method별 obs 형태가 달라서 분리
# ══════════════════════════════════════════════════════════════════════

def _episode_common_metrics(ep_info, ext, stale_hist, init_unc, u_pct,
                             ep_steps, ep_return, num_targets, drone_prox):
    unc_red       = (init_unc - u_pct) / max(init_unc, 1e-6) * 100.0
    prox_per_1000 = drone_prox / max(ep_steps / 1000.0, 1e-6)

    final_committed_count = int(ext["ext_committed_final"])
    final_stale_count     = int(stale_hist[-1]) if stale_hist else 0

    # avg_*: episode 전체 평균 (AUC 성격) / final_*: episode 마지막 시점 값
    avg_committed_ratio   = ext["ext_committed_avg"] / max(num_targets, 1)
    avg_stale_ratio       = ext["ext_stale_avg"]     / max(num_targets, 1)
    final_committed_ratio = final_committed_count / max(num_targets, 1)
    final_stale_ratio     = final_stale_count     / max(num_targets, 1)

    return {
        "uncertainty_reduction":  round(unc_red, 4),
        "final_uncertainty":      round(u_pct, 4),
        "final_mean_trP":         round(ext["ext_mean_trP_final"], 6),
        "auc_mean_trP":           round(ext["ext_auc_mean_trP"], 6),
        # committed_ratio / stale_ratio == final 기준 (혼동 방지를 위해 final_*로도 동시 저장)
        "committed_ratio":        round(final_committed_ratio, 6),
        "stale_ratio":            round(final_stale_ratio, 6),
        "final_committed_ratio":  round(final_committed_ratio, 6),
        "final_stale_ratio":      round(final_stale_ratio, 6),
        "avg_committed_ratio":    round(avg_committed_ratio, 6),
        "avg_stale_ratio":        round(avg_stale_ratio, 6),
        "time_under_good_track":  round(ext["ext_time_under_good_track"], 6),
        "prox_per_1000":          round(prox_per_1000, 6),
        "episode_return":         round(ep_return, 4),
        "episode_length":         ep_steps,
        "collision_count":        drone_prox,
        "terrain_hit_count":      ep_info.get("terrain_hits_ep", 0),
        "reacquire_count":        ext["ext_reacquire_count"],
        "final_committed_count":  final_committed_count,
        "final_stale_count":      final_stale_count,
    }


@torch.no_grad()
def run_one_episode_gat(env, base_env, agent, ext_tracker, eval_seed,
                         num_drones, num_targets, n_targets, nt_non_terrain,
                         target_types, acc_class):
    base_eval.set_seed_torch(eval_seed)
    obs = env.reset()
    acc = acc_class()
    last_info = {}
    ep_return = 0.0

    raw_u    = base_env.ds_map.total_uncertainty()
    init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

    ext_tracker.reset(base_eval._target_positions(base_env))
    trP_hist, commit_hist, stale_hist, tracked_hist = [], [], [], []
    T_commit_50, T_trP_below = [None], [None]

    step = 0
    for step in range(MAX_STEPS):
        acts, _, _ = agent.get_action_and_value(obs, deterministic=False)
        nobs, rews, dones, info = env.step(acts.cpu().numpy())
        acc.update(info, step); last_info = info
        ep_return += float(np.mean(rews))
        obs = nobs
        base_eval._run_ext_tracker_step(ext_tracker, base_env)
        base_eval._collect_ext(ext_tracker, step, n_targets,
                                trP_hist, commit_hist, stale_hist, tracked_hist,
                                T_commit_50, T_trP_below)
        if dones.all():
            break

    ep_steps = step + 1
    ep_info  = acc.summarize(last_info, nt_non_terrain)
    ext      = base_eval._summarize_ext(ext_tracker, trP_hist, commit_hist,
                                         stale_hist, tracked_hist,
                                         T_commit_50, T_trP_below,
                                         target_types, n_targets)
    u_pct      = ep_info.get("uncertainty_pct", float("nan"))
    drone_prox = ep_info.get("drone_colls_ep", 0)

    return _episode_common_metrics(ep_info, ext, stale_hist, init_unc, u_pct,
                                    ep_steps, ep_return, num_targets, drone_prox)


@torch.no_grad()
def run_one_episode_graph_token(env, base_env, agent, ext_tracker, eval_seed,
                                 num_drones, num_targets, n_targets, nt_non_terrain,
                                 target_types, acc_class):
    base_eval.set_seed_torch(eval_seed)
    obs, gobs = env.reset()
    acc = acc_class()
    last_info = {}
    ep_return = 0.0

    raw_u    = base_env.ds_map.total_uncertainty()
    init_unc = float(raw_u / max(nt_non_terrain, 1) * 100.0)

    ext_tracker.reset(base_eval._target_positions(base_env))
    trP_hist, commit_hist, stale_hist, tracked_hist = [], [], [], []
    T_commit_50, T_trP_below = [None], [None]

    step = 0
    for step in range(MAX_STEPS):
        acts, _, _, _, _ = agent.get_action_and_value(obs, gobs, deterministic=False)
        (nobs, ngobs), rews, dones, info = env.step(acts.cpu().numpy())
        acc.update(info, step); last_info = info
        ep_return += float(np.mean(rews))
        obs, gobs = nobs, ngobs
        base_eval._run_ext_tracker_step(ext_tracker, base_env)
        base_eval._collect_ext(ext_tracker, step, n_targets,
                                trP_hist, commit_hist, stale_hist, tracked_hist,
                                T_commit_50, T_trP_below)
        if dones.all():
            break

    ep_steps = step + 1
    ep_info  = acc.summarize(last_info, nt_non_terrain)
    ext      = base_eval._summarize_ext(ext_tracker, trP_hist, commit_hist,
                                         stale_hist, tracked_hist,
                                         T_commit_50, T_trP_below,
                                         target_types, n_targets)
    u_pct      = ep_info.get("uncertainty_pct", float("nan"))
    drone_prox = ep_info.get("drone_colls_ep", 0)

    return _episode_common_metrics(ep_info, ext, stale_hist, init_unc, u_pct,
                                    ep_steps, ep_return, num_targets, drone_prox)


EPISODE_RUNNERS = {
    "gat_mappo":   run_one_episode_gat,
    "graph_token": run_one_episode_graph_token,
}


# ══════════════════════════════════════════════════════════════════════
# 1 checkpoint 평가 (method x train_seed x scale) -> 100 episode
# ══════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(method, train_seed, checkpoint_path, scale_name,
                         n_episodes, out_dir: Path, device: str):
    from isaac_env_v13 import TargetBeliefTracker
    if method == "gat_mappo":
        from gat_mappo_v13 import EpisodeInfoAccumulator
    else:
        from graph_token_mappo_v13 import EpisodeInfoAccumulator

    per_ep_dir = out_dir / "per_episode"
    per_ep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_ep_dir / f"{method}_seed{train_seed}_{scale_name}.csv"

    done_eps = _get_done_episodes(csv_path)
    todo_eps = [i for i in range(n_episodes) if i not in done_eps]

    print(f"    [{method} seed{train_seed} {scale_name}] "
          f"done={len(done_eps)}/{n_episodes}  todo={len(todo_eps)}  -> {csv_path.name}")

    if not todo_eps:
        print("      -> already complete, skip")
        return

    base_env, env, num_drones, num_targets = build_env(method, scale_name)
    agent = load_policy(method, checkpoint_path, num_drones, device)

    target_types = tuple(int(t.value) if hasattr(t, "value") else int(t)
                          for t in base_env.target_types)
    n_targets      = base_env.num_targets
    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum())

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                       base_eval._target_positions(base_env))

    runner = EPISODE_RUNNERS[method]
    write_header = not csv_path.exists()

    from tqdm import tqdm
    bar = tqdm(todo_eps, ncols=140,
               desc=f"    {method:12s} seed{train_seed} {scale_name:6s}",
               bar_format="{desc}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in bar:
        eval_seed = make_eval_seed(scale_name, ep_idx)
        metrics = runner(env, base_env, agent, ext_tracker, eval_seed,
                          num_drones, num_targets, n_targets, nt_non_terrain,
                          target_types, EpisodeInfoAccumulator)

        row = {
            "method": method, "train_seed": train_seed, "scale": scale_name,
            "eval_episode": ep_idx, "eval_seed": eval_seed,
            "checkpoint_path": checkpoint_path,
        }
        row.update(metrics)
        _write_row(csv_path, row, write_header)
        write_header = False

        bar.set_postfix(ordered_dict={
            "U": f"{metrics['final_uncertainty']:.1f}%",
            "Cmt": f"{metrics['committed_ratio']:.2f}",
            "St": f"{metrics['stale_ratio']:.2f}",
        })


# ══════════════════════════════════════════════════════════════════════
# 집계 — seed-level, cross-seed, statistical test
# ══════════════════════════════════════════════════════════════════════

def load_all_per_episode(out_dir: Path) -> pd.DataFrame:
    per_ep_dir = out_dir / "per_episode"
    files = sorted(per_ep_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-episode CSVs found under {per_ep_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    out_path = out_dir / "per_episode_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df)} rows)")
    return df


def aggregate_per_seed(per_episode_df: pd.DataFrame, out_dir: Path,
                        expected_episodes: int) -> pd.DataFrame:
    """
    method x train_seed x scale 별로 100 episode 평균/표준편차.
    _std_episode 컬럼은 그 checkpoint 안에서의 episode-level variation이며
    논문 headline 통계로는 쓰지 않는다 (참고용).

    complete 컬럼: n_eval_episodes >= expected_episodes 인지 여부.
    평가가 중간에 끊긴(incomplete) group은 이 함수에서 걸러내지는 않고
    표시만 하며, aggregate_across_seeds / run_seed_level_tests에서
    complete==True인 group만 사용한다 (headline 통계에 부분 결과가
    섞이지 않도록).
    """
    rows = []
    group_cols = ["method", "train_seed", "scale"]
    for keys, g in per_episode_df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        n = len(g)
        row["n_eval_episodes"] = n
        row["complete"] = bool(n >= expected_episodes)
        for m in PER_EPISODE_METRIC_COLS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std_episode"] = g[m].std(ddof=1)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["method", "scale", "train_seed"])

    incomplete = df[~df["complete"]]
    if len(incomplete) > 0:
        print(f"\n  [WARN] {len(incomplete)} incomplete evaluation group(s) "
              f"(n_eval_episodes < {expected_episodes}):")
        print("  " + incomplete[["method", "train_seed", "scale", "n_eval_episodes"]]
              .to_string(index=False).replace("\n", "\n  "))
        print("  -> excluded from seed_level_summary.csv / seed_level_tests.csv\n")

    out_path = out_dir / "per_seed_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df)} rows, {len(df) - len(incomplete)} complete)")
    return df


def aggregate_across_seeds(per_seed_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    method x scale x metric 별로 5개 train_seed 평균의 mean +/- std
    (논문 headline: mean +/- std across N training seeds).

    per_seed_df의 complete==True인 group만 사용한다 (평가가 덜 끝난
    checkpoint가 headline 통계에 섞이지 않도록).
    """
    from scipy import stats as sstats

    complete_df = per_seed_df[per_seed_df["complete"]].copy()
    if len(complete_df) < len(per_seed_df):
        print(f"  [aggregate_across_seeds] using {len(complete_df)}/{len(per_seed_df)} "
              f"complete group(s) only")

    rows = []
    for (method, scale), g in complete_df.groupby(["method", "scale"]):
        n_seeds = len(g)
        for m in PER_EPISODE_METRIC_COLS:
            vals = g[f"{m}_mean"].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1)) if n_seeds > 1 else float("nan")
            se   = std / np.sqrt(n_seeds) if n_seeds > 1 else float("nan")
            if n_seeds > 1 and se == se:  # not nan
                tcrit = sstats.t.ppf(0.975, df=n_seeds - 1)
                ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
            else:
                ci_low, ci_high = float("nan"), float("nan")
            rows.append({
                "method": method, "scale": scale, "metric": m,
                "n_train_seeds": n_seeds,
                "mean_across_train_seeds": mean,
                "std_across_train_seeds": std,
                "se_across_train_seeds": se,
                "ci95_low": ci_low, "ci95_high": ci_high,
            })

    df = pd.DataFrame(rows).sort_values(["scale", "metric", "method"])
    out_path = out_dir / "seed_level_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df)} rows)")
    return df


def run_seed_level_tests(per_seed_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    scale x metric 별로 graph_token vs gat_mappo 의 5개 seed-level 값을
    train_seed로 짝지어(paired) Wilcoxon signed-rank test.

    n=5라 p-value 단독 해석은 위험하다는 점을 caption에 명시할 것 -
    여기서는 mean_diff / seed-level mean+std를 같이 저장한다.

    per_seed_df의 complete==True인 group만 사용하고, 그중에서도
    두 method 모두 값이 있는(=paired) train_seed만 비교한다.
    """
    from scipy.stats import wilcoxon

    complete_df = per_seed_df[per_seed_df["complete"]].copy()
    if len(complete_df) < len(per_seed_df):
        print(f"  [run_seed_level_tests] using {len(complete_df)}/{len(per_seed_df)} "
              f"complete group(s) only")

    rows = []
    scales = sorted(complete_df["scale"].unique())
    for scale in scales:
        sub = complete_df[complete_df["scale"] == scale]
        g_graph = sub[sub["method"] == "graph_token"].set_index("train_seed")
        g_gat   = sub[sub["method"] == "gat_mappo"].set_index("train_seed")
        common_seeds = sorted(set(g_graph.index) & set(g_gat.index))
        if len(common_seeds) < 2:
            print(f"  [WARN] scale={scale}: matched train_seed < 2, skip test")
            continue

        for m in PER_EPISODE_METRIC_COLS:
            col = f"{m}_mean"
            graph_vals = g_graph.loc[common_seeds, col].to_numpy(dtype=float)
            gat_vals   = g_gat.loc[common_seeds, col].to_numpy(dtype=float)

            diff = graph_vals - gat_vals
            mean_diff = float(np.mean(diff))

            p_value = float("nan")
            try:
                _, p_value = wilcoxon(graph_vals, gat_vals,
                                       zero_method="wilcox", alternative="two-sided")
            except ValueError as e:
                print(f"  [WARN] wilcoxon failed for scale={scale} metric={m}: {e}")

            higher_better = METRIC_DIRECTION_HIGHER_BETTER.get(m)
            if higher_better is None:
                better_method = "n/a (direction unspecified)"
            elif mean_diff == 0:
                better_method = "tie"
            elif (mean_diff > 0) == higher_better:
                better_method = "graph_token"
            else:
                better_method = "gat_mappo"

            rows.append({
                "scale": scale, "metric": m, "n_seeds_matched": len(common_seeds),
                "graph_mean": float(np.mean(graph_vals)),
                "graph_std":  float(np.std(graph_vals, ddof=1)) if len(graph_vals) > 1 else float("nan"),
                "gat_mean":   float(np.mean(gat_vals)),
                "gat_std":    float(np.std(gat_vals, ddof=1)) if len(gat_vals) > 1 else float("nan"),
                "mean_diff_graph_minus_gat": mean_diff,
                "wilcoxon_p": p_value,
                "better_method": better_method,
            })

    df = pd.DataFrame(rows).sort_values(["scale", "metric"])
    out_path = out_dir / "seed_level_tests.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df)} rows)")
    print("  [NOTE] n=5 seed-level test - report effect size (mean_diff) and "
          "seed-level mean+/-std alongside p-value, not p-value alone.")
    return df


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Multi-seed robustness evaluation")
    ap.add_argument("--root",     type=str, default=str(REPO_ROOT))
    ap.add_argument("--run_root", type=str, default=str(REPO_ROOT / "seed_robustness_runs"),
                     help="training seed checkpoint root "
                          "(run_root/{method}/stage1/seed{N}/*.pt)")
    ap.add_argument("--out_dir",  type=str, default=str(REPO_ROOT / "seed_robustness_eval"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--checkpoint_ep", type=int, default=1500,
                     help="which stage1 checkpoint episode to load "
                          "(e.g. 1500 -> graph_token_ep01500.pt)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds",   type=int, nargs="+", default=TRAIN_SEEDS_DEFAULT)
    ap.add_argument("--scales",  type=str, nargs="+", default=SCALE_ORDER,
                     choices=SCALE_ORDER)
    ap.add_argument("--methods", type=str, nargs="+",
                     default=list(METHOD_SPECS.keys()),
                     choices=list(METHOD_SPECS.keys()))
    ap.add_argument("--restart", action="store_true",
                     help="delete existing per-episode CSVs for the selected "
                          "method/seed/scale combos and re-evaluate from scratch")
    ap.add_argument("--aggregate_only", action="store_true",
                     help="skip evaluation, only (re)build summary/test CSVs "
                          "from existing per_episode/*.csv")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[MULTI-SEED ROBUSTNESS EVAL]  Graph-Token MAPPO vs GAT-MAPPO")
    print(f"  run_root : {run_root}")
    print(f"  out_dir  : {out_dir}")
    print(f"  seeds    : {args.seeds}")
    print(f"  scales   : {args.scales}")
    print(f"  methods  : {args.methods}")
    print(f"  episodes : {args.episodes} per checkpoint")
    print(f"  ckpt_ep  : {args.checkpoint_ep}")
    print(f"  device   : {args.device}")
    print("=" * 80)

    if not args.aggregate_only:
        total_combos = len(args.methods) * len(args.seeds) * len(args.scales)
        combo_idx = 0
        for method in args.methods:
            spec = METHOD_SPECS[method]
            for seed in args.seeds:
                ckpt_path = (run_root / spec["ckpt_dirname"] / "stage1" / f"seed{seed}"
                             / f"{spec['ckpt_prefix']}{args.checkpoint_ep:05d}.pt")

                if not ckpt_path.exists():
                    print(f"\n  [SKIP] {method} seed={seed}: checkpoint not found")
                    print(f"         {ckpt_path}")
                    combo_idx += len(args.scales)
                    continue

                for scale in args.scales:
                    combo_idx += 1
                    print(f"\n[{combo_idx}/{total_combos}] {method} seed={seed} scale={scale}")

                    if args.restart:
                        csv_path = out_dir / "per_episode" / f"{method}_seed{seed}_{scale}.csv"
                        if csv_path.exists():
                            csv_path.unlink()
                            print(f"  [restart] removed {csv_path.name}")

                    evaluate_checkpoint(method, seed, str(ckpt_path), scale,
                                         args.episodes, out_dir, args.device)

    print("\n" + "=" * 80)
    print("[AGGREGATION]")
    print("=" * 80)
    per_episode_df = load_all_per_episode(out_dir)
    per_seed_df    = aggregate_per_seed(per_episode_df, out_dir,
                                         expected_episodes=args.episodes)
    aggregate_across_seeds(per_seed_df, out_dir)
    run_seed_level_tests(per_seed_df, out_dir)

    print("\nAll done.")
    print(f"Results under: {out_dir}")
    print("  per_episode_metrics.csv  - raw, transparency/diagnostics only")
    print("  per_seed_summary.csv     - 100-episode mean per checkpoint")
    print("  seed_level_summary.csv   - mean +/- std across training seeds (paper headline)")
    print("  seed_level_tests.csv     - paired Wilcoxon, graph_token vs gat_mappo")


if __name__ == "__main__":
    main()
