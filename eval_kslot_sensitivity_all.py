"""
eval_kslot_sensitivity_all.py
==============================
Graph-Token MAPPO: active-track token budget K sensitivity 평가.

목적
  K = 8 / 12(default) / 16 각각 독립적으로 학습된 checkpoint를
  (4 UAV/20 targets, 1500 episodes, seed 0, stage1) 3개 scale
  (4x20 / 8x40 / 12x60)에서 100 stochastic episode씩 평가하고,
  K가 커질/작아질 때 tracking 성능이 어떻게 바뀌는지 본다.

  이 실험은 "K=12가 통계적으로 우월하다"를 증명하는 seed-robustness
  실험이 아니다. 각 K는 training seed 1개짜리 단일 checkpoint이므로,
  결과는 fixed-checkpoint sensitivity diagnostic으로만 취급한다.
  (episode-level paired test는 가능하지만 training-run variability는
  담지 못한다.)

재사용 (새로 정의하지 않음)
  - env / external tracker : eval_v13_scale_all.py의 _make_env_v13,
    _target_positions, _run_ext_tracker_step, _collect_ext, _summarize_ext,
    set_seed_torch, SCALE_CONFIGS, MAX_STEPS
  - episode 실행 / metric 정의 : eval_seed_robustness_all.py의
    run_one_episode_graph_token (내부에서 _episode_common_metrics 사용),
    PER_EPISODE_METRIC_COLS, _get_done_episodes
  두 파일 다 import해서 그대로 쓴다. metric 정의가 main experiment /
  seed-robustness 실험과 어긋나지 않도록 하기 위함이며, committed/stale
  비율도 K-selected token이 아니라 external TargetBeliefTracker가 추적하는
  전체 target 기준으로 계산된다 (K=8이라고 8개 target만 평가하지 않음).

K별 checkpoint / 코드 매핑
  K=8  -> graph_token_mappo_v13_k8.py  (K_ACTIVE=8,  OBS_DIM=460)
  K=12 -> graph_token_mappo_v13_k12.py (K_ACTIVE=12, OBS_DIM=484, default)
  K=16 -> graph_token_mappo_v13_k16.py (K_ACTIVE=16, OBS_DIM=508)
  checkpoint: kslot_runs/graph_token_k{K}/stage1/seed0/graph_token_ep{ep:05d}.pt

  K별 module은 importlib로 동적 로드한다 (고정 import 금지). K=8 checkpoint를
  K=12용 architecture에 로드하면 안 되므로, TrainConfig/MAPPOAgent/
  GraphTokenObsWrapper/EpisodeInfoAccumulator를 전부 해당 K의 모듈에서 가져온다.

평가 프로토콜
  - stochastic action sampling (deterministic=False) — main/seed-robustness와 동일.
  - eval_seed = 20000 + scale_id*1000 + episode_idx
    -> 같은 (scale, episode_idx)면 K값과 무관하게 동일 eval_seed.
       (seed-robustness 실험의 10000 base와 겹치지 않도록 20000 사용)

통계
  각 (scale, metric)에서 K8 vs K12 vs K16 을 eval_episode로 짝지어
  Friedman test -> pairwise paired Wilcoxon (K8-K12, K12-K16, K8-K16) ->
  Holm-Bonferroni 보정. training seed가 K당 1개뿐이므로 diagnostic 용도로만
  해석한다 (코드/출력에 명시).

Resume
  per_episode/k{K}_{scale}.csv 의 eval_episode 기준으로 이어서 진행.

실행
    cd /path/to/repository
    conda activate drone_env_310
    python eval_kslot_sensitivity_all.py ^
      --root . ^
      --run_root ./kslot_runs ^
      --out_dir ./kslot_eval ^
      --episodes 100 ^
      --checkpoint_ep 1500 ^
      --device cuda ^
      --ks 8 12 16 ^
      --scales 4x20 8x40 12x60

(인자를 생략하면 위 값들은 모두 repository root 기준 기본값으로 자동 설정된다.)

집계만 다시:
    python eval_kslot_sensitivity_all.py --out_dir ./kslot_eval --aggregate_only
"""
import os
import sys
import csv
import importlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

# env/tracker/metric 로직 재사용 (새로 정의하지 않음)
import eval_v13_scale_all as base_eval
import eval_seed_robustness_all as seed_eval

MAX_STEPS            = base_eval.MAX_STEPS
SCALE_CONFIGS        = base_eval.SCALE_CONFIGS
SCALE_ORDER          = ["4x20", "8x40", "12x60"]
PER_EPISODE_METRIC_COLS = seed_eval.PER_EPISODE_METRIC_COLS  # committed/stale는 이미
                                                              # external tracker 기준 전체 target
run_one_episode_graph_token = seed_eval.run_one_episode_graph_token
_get_done_episodes           = seed_eval._get_done_episodes

TRAIN_SEED = 0  # K=8/12/16 각각 training seed 1개짜리 checkpoint

K_SPECS = {
    8:  {"module": "graph_token_mappo_v13_k8",  "run_dir": "graph_token_k8"},
    12: {"module": "graph_token_mappo_v13_k12", "run_dir": "graph_token_k12"},
    16: {"module": "graph_token_mappo_v13_k16", "run_dir": "graph_token_k16"},
}
K_ORDER_DEFAULT = [8, 12, 16]

CSV_COLUMNS = (
    ["k", "scale", "eval_episode", "eval_seed", "checkpoint_path"]
    + PER_EPISODE_METRIC_COLS
)


# ══════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════

def make_eval_seed(scale_name: str, episode_idx: int) -> int:
    scale_id = SCALE_ORDER.index(scale_name)
    return 20000 + scale_id * 1000 + episode_idx


def get_checkpoint_path(run_root: Path, k: int, ep: int) -> Path:
    spec = K_SPECS[k]
    return (run_root / spec["run_dir"] / "stage1" / f"seed{TRAIN_SEED}"
            / f"graph_token_ep{ep:05d}.pt")


def _write_row(csv_path: Path, row: dict, write_header: bool):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


# ══════════════════════════════════════════════════════════════════════
# K별 env / policy 로딩 — 반드시 K에 맞는 module + k_active 사용
# ══════════════════════════════════════════════════════════════════════

def build_env_and_agent(k: int, scale_name: str, checkpoint_path: str, device: str):
    """
    K에 대응하는 graph_token_mappo_v13_k{K}.py 모듈을 동적 import해서
    그 모듈의 TrainConfig/MAPPOAgent/GraphTokenObsWrapper/
    EpisodeInfoAccumulator를 사용한다. 절대 고정된
    `from graph_token_mappo_v13 import ...`를 쓰지 않는다 — K별로
    OBS_DIM/K_ACTIVE/token 구성이 다르기 때문에 architecture가 안 맞으면
    checkpoint load가 깨지거나(잘못하면 조용히 잘못된 평가가 됨).
    """
    module = importlib.import_module(K_SPECS[k]["module"])

    cfg_scale = SCALE_CONFIGS[scale_name]
    num_drones, num_targets = cfg_scale["num_drones"], cfg_scale["num_targets"]

    # env도 반드시 이 K로 만든다 (모든 K에서 k_active=12로 만들면 절대 안 됨)
    base_env, _ = base_eval._make_env_v13(num_drones, num_targets, k_active=k)
    env = module.GraphTokenObsWrapper(base_env)

    cfg = module.TrainConfig()
    cfg.device = device
    agent = module.MAPPOAgent(cfg)
    agent.load(checkpoint_path)
    agent.actor.eval(); agent.critic.eval()
    agent.ep_prog = 1.0

    return base_env, env, agent, module.EpisodeInfoAccumulator, num_drones, num_targets


# ══════════════════════════════════════════════════════════════════════
# 1 (K, scale) 평가 -> 100 episode
# ══════════════════════════════════════════════════════════════════════

def evaluate_one_k_scale(k: int, checkpoint_path: str, scale_name: str,
                          n_episodes: int, out_dir: Path, device: str):
    from isaac_env_v13 import TargetBeliefTracker

    per_ep_dir = out_dir / "per_episode"
    per_ep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_ep_dir / f"k{k}_{scale_name}.csv"

    done_eps = _get_done_episodes(csv_path)
    todo_eps = [i for i in range(n_episodes) if i not in done_eps]

    print(f"    [K={k} {scale_name}] done={len(done_eps)}/{n_episodes}  "
          f"todo={len(todo_eps)}  -> {csv_path.name}")

    if not todo_eps:
        print("      -> already complete, skip")
        return

    base_env, env, agent, EpisodeInfoAccumulator, num_drones, num_targets = \
        build_env_and_agent(k, scale_name, checkpoint_path, device)

    target_types = tuple(int(t.value) if hasattr(t, "value") else int(t)
                          for t in base_env.target_types)
    n_targets      = base_env.num_targets
    nt_non_terrain = int((~base_env.ds_map.terrain_mask).sum())

    ext_tracker = TargetBeliefTracker(n_targets, target_types,
                                       base_eval._target_positions(base_env))

    write_header = not csv_path.exists()

    from tqdm import tqdm
    bar = tqdm(todo_eps, ncols=140, desc=f"    K={k:<2d} {scale_name:6s}",
               bar_format="{desc}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")

    for ep_idx in bar:
        eval_seed = make_eval_seed(scale_name, ep_idx)
        # committed/stale은 run_one_episode_graph_token 내부의
        # _episode_common_metrics()가 external tracker(전체 target 기준)로 계산.
        # K-selected token 개수만큼만 계산하지 않는다.
        metrics = run_one_episode_graph_token(
            env, base_env, agent, ext_tracker, eval_seed,
            num_drones, num_targets, n_targets, nt_non_terrain,
            target_types, EpisodeInfoAccumulator)

        row = {
            "k": k, "scale": scale_name, "eval_episode": ep_idx,
            "eval_seed": eval_seed, "checkpoint_path": str(checkpoint_path),
        }
        row.update(metrics)
        _write_row(csv_path, row, write_header)
        write_header = False

        bar.set_postfix(ordered_dict={
            "U": f"{metrics['final_uncertainty']:.1f}%",
            "AvgCmt": f"{metrics['avg_committed_ratio']:.2f}",
            "AvgSt":  f"{metrics['avg_stale_ratio']:.2f}",
        })


# ══════════════════════════════════════════════════════════════════════
# 집계 — training seed가 K당 1개뿐이므로 across-seed 통계는 만들지 않는다.
# 각 K x scale 의 100 episode에 대한 mean/std 만 낸다.
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


def aggregate_kslot(per_episode_df: pd.DataFrame, out_dir: Path,
                     expected_episodes: int) -> pd.DataFrame:
    """
    K x scale 별 100-episode mean +/- std.
    complete 컬럼: n_eval_episodes >= expected_episodes.
    (seed-robustness와 달리 training seed가 1개뿐이라 "across-seed" 개념이 없음)
    """
    rows = []
    for (k, scale), g in per_episode_df.groupby(["k", "scale"]):
        n = len(g)
        row = {"k": k, "scale": scale, "n_eval_episodes": n,
               "complete": bool(n >= expected_episodes)}
        for m in PER_EPISODE_METRIC_COLS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"]  = g[m].std(ddof=1)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["scale", "k"])

    incomplete = df[~df["complete"]]
    if len(incomplete) > 0:
        print(f"\n  [WARN] {len(incomplete)} incomplete (K, scale) group(s) "
              f"(n_eval_episodes < {expected_episodes}):")
        print("  " + incomplete[["k", "scale", "n_eval_episodes"]]
              .to_string(index=False).replace("\n", "\n  "))
        print("  -> excluded from kslot_pairwise_tests.csv\n")

    out_path = out_dir / "kslot_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df)} rows, {len(df) - len(incomplete)} complete)")
    return df


# ══════════════════════════════════════════════════════════════════════
# 통계 — episode-level paired diagnostic
#   Friedman(K8, K12, K16) -> pairwise paired Wilcoxon -> Holm-Bonferroni
#
# NOTE: These episode-level paired tests are diagnostic only because
#       each K configuration was trained with a single training seed.
#       They do not quantify training-run variability.
# ══════════════════════════════════════════════════════════════════════

def _holm_bonferroni(pvals):
    """Holm step-down 보정. 입력 순서를 유지해서 반환."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((n - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def run_kslot_tests(per_episode_df: pd.DataFrame, kslot_summary_df: pd.DataFrame,
                     out_dir: Path, ks) -> pd.DataFrame:
    from scipy.stats import friedmanchisquare, wilcoxon

    pair_labels = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]

    rows = []
    scales = sorted(per_episode_df["scale"].unique())
    for scale in scales:
        complete_ks = set(
            kslot_summary_df[(kslot_summary_df["scale"] == scale)
                              & (kslot_summary_df["complete"])]["k"].tolist()
        )
        if not set(ks).issubset(complete_ks):
            missing = set(ks) - complete_ks
            print(f"  [WARN] scale={scale}: K={sorted(missing)} incomplete, skip tests")
            continue

        sub = per_episode_df[per_episode_df["scale"] == scale]
        by_k = {k: sub[sub["k"] == k].set_index("eval_episode") for k in ks}
        common_eps = sorted(set.intersection(*[set(d.index) for d in by_k.values()]))
        if len(common_eps) < 2:
            print(f"  [WARN] scale={scale}: matched episodes < 2, skip tests")
            continue

        for m in PER_EPISODE_METRIC_COLS:
            vals = {k: by_k[k].loc[common_eps, m].to_numpy(dtype=float) for k in ks}

            friedman_p = float("nan")
            if len(ks) >= 3:
                try:
                    _, friedman_p = friedmanchisquare(*[vals[k] for k in ks])
                except ValueError as e:
                    print(f"  [WARN] friedman failed scale={scale} metric={m}: {e}")

            raw_p = []
            pair_stats = []
            for a, b in pair_labels:
                diff = vals[a] - vals[b]
                mean_diff = float(np.mean(diff))
                p = float("nan")
                try:
                    _, p = wilcoxon(vals[a], vals[b],
                                     zero_method="wilcox", alternative="two-sided")
                except ValueError as e:
                    print(f"  [WARN] wilcoxon failed scale={scale} metric={m} "
                          f"K{a}-K{b}: {e}")
                raw_p.append(p)
                pair_stats.append({
                    "comparison": f"K{a}_vs_K{b}",
                    "mean_a": float(np.mean(vals[a])),
                    "mean_b": float(np.mean(vals[b])),
                    "mean_difference": mean_diff,
                    "wilcoxon_p_raw": p,
                })

            valid_mask = [p == p for p in raw_p]  # not NaN
            adj_p = np.full(len(raw_p), float("nan"))
            if any(valid_mask):
                valid_idx = [i for i, v in enumerate(valid_mask) if v]
                adj_valid = _holm_bonferroni([raw_p[i] for i in valid_idx])
                for j, i in enumerate(valid_idx):
                    adj_p[i] = adj_valid[j]

            for i, ps in enumerate(pair_stats):
                rows.append({
                    "scale": scale, "metric": m,
                    "n_episodes_matched": len(common_eps),
                    "friedman_p": friedman_p,
                    **ps,
                    "wilcoxon_p_holm": adj_p[i],
                })

    df = pd.DataFrame(rows).sort_values(["scale", "metric", "comparison"])
    out_path = out_dir / "kslot_pairwise_tests.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df)} rows)")
    print("  [NOTE] These episode-level paired tests are diagnostic only because")
    print("         each K configuration was trained with a single training seed.")
    print("         They do not quantify training-run variability.")
    return df


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Graph-Token MAPPO K-slot sensitivity evaluation")
    ap.add_argument("--root",     type=str, default=str(REPO_ROOT))
    ap.add_argument("--run_root", type=str, default=str(REPO_ROOT / "kslot_runs"),
                     help="K-slot checkpoint root "
                          "(run_root/graph_token_k{K}/stage1/seed0/*.pt)")
    ap.add_argument("--out_dir",  type=str, default=str(REPO_ROOT / "kslot_eval"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--checkpoint_ep", type=int, default=1500,
                     help="stage1 checkpoint episode to load "
                          "(e.g. 1500 -> graph_token_ep01500.pt)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ks",      type=int, nargs="+", default=K_ORDER_DEFAULT,
                     choices=list(K_SPECS.keys()))
    ap.add_argument("--scales",  type=str, nargs="+", default=SCALE_ORDER,
                     choices=SCALE_ORDER)
    ap.add_argument("--restart", action="store_true",
                     help="delete existing per-episode CSVs for the selected "
                          "K/scale combos and re-evaluate from scratch")
    ap.add_argument("--aggregate_only", action="store_true",
                     help="skip evaluation, only (re)build summary/test CSVs "
                          "from existing per_episode/*.csv")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(args.ks)

    print("=" * 80)
    print("[K-SLOT SENSITIVITY EVAL]  Graph-Token MAPPO  K = " + ", ".join(str(k) for k in ks))
    print(f"  run_root : {run_root}")
    print(f"  out_dir  : {out_dir}")
    print(f"  scales   : {args.scales}")
    print(f"  episodes : {args.episodes} per (K, scale)")
    print(f"  ckpt_ep  : {args.checkpoint_ep}")
    print(f"  device   : {args.device}")
    print(f"  train_seed (fixed, 1 per K) : {TRAIN_SEED}")
    print("=" * 80)

    if not args.aggregate_only:
        total_combos = len(ks) * len(args.scales)
        combo_idx = 0
        for k in ks:
            ckpt_path = get_checkpoint_path(run_root, k, args.checkpoint_ep)

            if not ckpt_path.exists():
                print(f"\n  [SKIP] K={k}: checkpoint not found")
                print(f"         {ckpt_path}")
                combo_idx += len(args.scales)
                continue

            for scale in args.scales:
                combo_idx += 1
                print(f"\n[{combo_idx}/{total_combos}] K={k} scale={scale}")

                if args.restart:
                    csv_path = out_dir / "per_episode" / f"k{k}_{scale}.csv"
                    if csv_path.exists():
                        csv_path.unlink()
                        print(f"  [restart] removed {csv_path.name}")

                evaluate_one_k_scale(k, str(ckpt_path), scale,
                                      args.episodes, out_dir, args.device)

    print("\n" + "=" * 80)
    print("[AGGREGATION]")
    print("=" * 80)
    per_episode_df   = load_all_per_episode(out_dir)
    kslot_summary_df = aggregate_kslot(per_episode_df, out_dir,
                                        expected_episodes=args.episodes)
    run_kslot_tests(per_episode_df, kslot_summary_df, out_dir, ks)

    print("\nAll done.")
    print(f"Results under: {out_dir}")
    print("  per_episode_metrics.csv    - raw, transparency/diagnostics only")
    print("  kslot_summary.csv          - 100-episode mean +/- std per (K, scale)")
    print("  kslot_pairwise_tests.csv   - Friedman + paired Wilcoxon (Holm-corrected),")
    print("                               fixed-checkpoint diagnostic only (1 seed per K)")


if __name__ == "__main__":
    main()
