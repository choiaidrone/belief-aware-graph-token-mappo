"""
analyze_results_v13.py — Scale Experiment v13 결과 분석
========================================================
[비교 대상]  5 method × 3 scale = 15개 CSV
    GAT         (Zhao et al. 2025)   — gat_mappo_v13
    CC-MPSO     (2024)               — heuristics_v13
    SSM         (2023)               — heuristics_v13
    Random walk                      — eval 내 인라인
    Graph-Token (Proposed)           — graph_token_mappo_v13

[CSV 경로]
    <repository root>/eval_results/v13/scale_{scale}_{method}.csv

[출력]
    <repository root>/eval_results/v13/analysis/
        comparison_table_{scale}.csv     ← scale별 논문 Table용
        comparison_table_all.csv         ← 전체 합산 (4x20만)
        fig_search.png                   ← 탐색 성능
        fig_tracking.png                 ← 추적 품질
        fig_trP_type.png                 ← 타입별 tr(P)
        fig_safety.png                   ← Safety / Coordination
        fig_scale.png                    ← Scale 확장성
        fig_summary.png                  ← 핵심 지표 한눈에

[실행법]
    python analyze_results_v13.py
    python analyze_results_v13.py --scales 4x20
    python analyze_results_v13.py --no_plot
"""

import os, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

IN_DIR  = str(REPO_ROOT / "eval_results" / "v13")
OUT_DIR = str(REPO_ROOT / "eval_results" / "v13" / "analysis")

SCALES = ["4x20", "8x40", "12x60"]

# ── 모델 정의 ──────────────────────────────────────────────────────────
MODELS = [
    {"key": "random",      "label": "Random",       "color": "#8B8B8B", "marker": "s", "ls": ":"},
    {"key": "ssm",         "label": "SSM (2023)",   "color": "#F4A300", "marker": "^", "ls": "--"},
    {"key": "ccmpso",      "label": "CC-MPSO (2024)","color": "#3A86FF","marker": "o", "ls": "--"},
    {"key": "gat",         "label": "GAT (2025)",   "color": "#9B59B6", "marker": "D", "ls": "-."},
    {"key": "graph_token", "label": "Proposed",     "color": "#E84393", "marker": "*", "ls": "-"},
]

GOOD_THRESH = 0.6

# ── 지표 정의 ──────────────────────────────────────────────────────────
# (컬럼명, 표시명, better방향)
METRICS_SEARCH = [
    ("unc_reduction_pct",         "Uncertainty Reduction %  ↑",  "↑"),
    ("uncertainty_pct",           "Final Uncertainty %  ↓",      "↓"),
]
METRICS_TRACKING = [
    ("ext_mean_trP_final",        "Mean tr(P) final  ↓",         "↓"),
    ("ext_auc_mean_trP",          "AUC mean tr(P)  ↓",           "↓"),
    ("committed_ratio",           "Committed ratio  ↑",          "↑"),
    ("stale_ratio_norm",          "Stale ratio  ↓",              "↓"),
    ("ext_T_commit_50",           "T_commit 50% (steps)  ↓",     "↓"),
    ("ext_time_under_good_track", "Time under good track  ↑",    "↑"),
    ("ext_reacquire_count",       "Reacquire count  ↑",          "↑"),
]
METRICS_TYPE = [
    ("ext_trP_infantry",  "tr(P) Infantry  ↓",  "↓"),
    ("ext_trP_tank",      "tr(P) Tank  ↓",      "↓"),
    ("ext_trP_artillery", "tr(P) Artillery  ↓", "↓"),
]
METRICS_SAFETY = [
    ("drone_prox_events", "Inter-UAV Proximity Events  ↓", "↓"),   # Safety
    ("active_at_end",     "Active Drones at End  ↑",       "↑"),   # Coordination
    ("mean_battery",      "Battery Remaining  ↑",          "↑"),
    ("rtb_count",         "RTB Count  ↓",                  "↓"),
    ("depleted_ep",       "Battery Depleted  ↓",           "↓"),
]
METRICS_ALL = METRICS_SEARCH + METRICS_TRACKING + METRICS_TYPE + METRICS_SAFETY


# ══════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════

def load_scale(scale: str) -> dict:
    """scale별 {method_key: DataFrame} 딕셔너리 반환."""
    dfs = {}
    for m in MODELS:
        path = os.path.join(IN_DIR, f"scale_{scale}_{m['key']}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dfs[m["key"]] = df
                print(f"    [{scale}] {m['label']:18s}: {len(df):3d} eps")
            except Exception as e:
                print(f"    [{scale}] {m['label']:18s}: 읽기 실패 ({e})")
        else:
            print(f"    [{scale}] {m['label']:18s}: 파일 없음 ({path})")
    return dfs


def load_all_scales(scales) -> dict:
    """전체 {scale: {method: df}} 구조 반환."""
    all_dfs = {}
    for scale in scales:
        print(f"\n  [로드] scale={scale}")
        all_dfs[scale] = load_scale(scale)
    return all_dfs


def _get(df, col):
    if df is None or col not in df.columns:
        return np.array([])
    return df[col].dropna().values


# ══════════════════════════════════════════════════════════════════════
# 비교표 생성
# ══════════════════════════════════════════════════════════════════════

def make_comparison_table(dfs: dict, scale: str) -> pd.DataFrame:
    """dfs: {method_key: df}  →  비교표 DataFrame."""
    try:
        from scipy.stats import f_oneway
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    rows = []
    for col, label, better in METRICS_ALL:
        row = {"Scale": scale, "Metric": label, "Better": better}
        vals_dict = {}
        for m in MODELS:
            arr = _get(dfs.get(m["key"]), col)
            vals_dict[m["key"]] = arr
            row[f"{m['label']} Mean"] = f"{arr.mean():.4f}" if len(arr) else "N/A"
            row[f"{m['label']} Std"]  = f"{arr.std():.4f}"  if len(arr) else "N/A"

        # 최고 모델
        best_key = None; best_val = None
        for m in MODELS:
            arr = vals_dict[m["key"]]
            if len(arr) == 0: continue
            v = arr.mean()
            if best_val is None: best_val = v; best_key = m["key"]
            elif better == "↑" and v > best_val: best_val = v; best_key = m["key"]
            elif better == "↓" and v < best_val: best_val = v; best_key = m["key"]
        row["Best"] = next((m["label"] for m in MODELS if m["key"] == best_key), "N/A")

        # ANOVA
        if HAS_SCIPY:
            groups = [vals_dict[m["key"]] for m in MODELS if len(vals_dict[m["key"]]) >= 2]
            if len(groups) >= 2:
                try:
                    _, p = f_oneway(*groups)
                    row["ANOVA p"] = f"{p:.4f}"; row["Sig"] = "✓" if p < 0.05 else ""
                except Exception:
                    row["ANOVA p"] = "N/A"; row["Sig"] = ""
            else:
                row["ANOVA p"] = "N/A"; row["Sig"] = ""
        else:
            row["ANOVA p"] = "N/A"; row["Sig"] = ""

        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 콘솔 요약 출력
# ══════════════════════════════════════════════════════════════════════

def print_summary(all_dfs: dict, scales: list):
    PRINT_COLS = [
        ("unc_reduction_pct",         "Uncertainty Reduction %"),
        ("uncertainty_pct",           "Final Uncertainty %"),
        ("ext_auc_mean_trP",          "AUC mean tr(P)"),
        ("committed_ratio",           "Committed ratio"),
        ("stale_ratio_norm",          "Stale ratio"),
        ("ext_time_under_good_track", "Time good track"),
        ("ext_trP_infantry",          "tr(P) Infantry"),
        ("ext_trP_tank",              "tr(P) Tank"),
        ("ext_trP_artillery",         "tr(P) Artillery"),
        ("drone_prox_events",         "Proximity Events"),
        ("active_at_end",             "Active at end"),
        ("mean_battery",              "Battery remaining"),
    ]
    col_w = 22

    for scale in scales:
        dfs = all_dfs.get(scale, {})
        if not dfs: continue
        header = f"\n{'─'*100}\n  Scale: {scale}\n{'─'*100}"
        print(header)
        hdr = f"  {'Metric':28s}"
        for m in MODELS: hdr += f"  {m['label']:>{col_w}s}"
        print(hdr); print(f"  {'─'*28}" + f"  {'─'*col_w}" * len(MODELS))

        for col, label in PRINT_COLS:
            line = f"  {label:28s}"
            for m in MODELS:
                arr = _get(dfs.get(m["key"]), col)
                if len(arr): line += f"  {arr.mean():>{col_w-6}.4f}±{arr.std():.3f}"
                else:        line += f"  {'N/A':>{col_w}s}"
            print(line)
    print(f"  {'─'*100}")


# ══════════════════════════════════════════════════════════════════════
# 시각화 헬퍼
# ══════════════════════════════════════════════════════════════════════

def _violin(ax, col, title, ylabel, dfs):
    data = []; labels = []; colors = []
    for m in MODELS:
        arr = _get(dfs.get(m["key"]), col)
        if len(arr) > 0:
            data.append(arr); labels.append(m["label"]); colors.append(m["color"])
    if not data: ax.set_visible(False); return
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.65)
    parts["cmeans"].set_color("black"); parts["cmeans"].set_linewidth(1.8)
    parts["cmedians"].set_color("red"); parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", length=0)


def _bar(ax, col, title, ylabel, dfs):
    vals=[]; errs=[]; labels=[]; colors=[]
    for m in MODELS:
        arr = _get(dfs.get(m["key"]), col)
        if len(arr) > 0:
            vals.append(arr.mean()); errs.append(arr.std())
            labels.append(m["label"]); colors.append(m["color"])
    if not vals: ax.set_visible(False); return
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, alpha=0.75, width=0.6, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="#333",
                capsize=4, linewidth=1.0, capthick=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def _legend_patches():
    import matplotlib.patches as mpatches
    return [mpatches.Patch(color=m["color"], alpha=0.75, label=m["label"])
            for m in MODELS]


def _save(fig, out_dir, fname):
    import matplotlib.pyplot as plt
    fig.legend(handles=_legend_patches(), loc="lower center", ncol=5,
               fontsize=8.5, frameon=True, framealpha=0.9,
               edgecolor="#CCC", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [저장] {fname}")


# ══════════════════════════════════════════════════════════════════════
# 그래프 생성
# ══════════════════════════════════════════════════════════════════════

def make_plots(all_dfs: dict, scales: list, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # 4x20 기준 단면 분석 (논문 main result)
    dfs_main = all_dfs.get("4x20", {})

    # ── 1. Search ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Search / Uncertainty Reduction  [4×20, N=100 ep]", fontweight="bold")
    _violin(axes[0], "unc_reduction_pct", "Uncertainty Reduction %  ↑", "%", dfs_main)
    _violin(axes[1], "uncertainty_pct",   "Final Uncertainty %  ↓",     "%", dfs_main)
    _save(fig, out_dir, "fig_search.png")

    # ── 2. Tracking ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Tracking Quality — Evaluation Tracker  [4×20, N=100 ep]", fontweight="bold")
    ax_f = axes.flatten()
    _violin(ax_f[0], "ext_mean_trP_final",        "Mean tr(P) final  ↓",       "tr(P)", dfs_main)
    _violin(ax_f[1], "ext_auc_mean_trP",           "AUC mean tr(P)  ↓",         "tr(P)", dfs_main)
    _violin(ax_f[2], "ext_time_under_good_track",  "Time under good track  ↑",  "ratio", dfs_main)
    _violin(ax_f[3], "committed_ratio",            "Committed ratio  ↑",        "ratio", dfs_main)
    _violin(ax_f[4], "stale_ratio_norm",           "Stale ratio  ↓",            "ratio", dfs_main)
    _violin(ax_f[5], "ext_reacquire_count",        "Reacquire count  ↑",        "count", dfs_main)
    _save(fig, out_dir, "fig_tracking.png")

    # ── 3. Target type별 tr(P) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Tracking Quality by Target Type — tr(P) ↓  [4×20, N=100 ep]", fontweight="bold")
    _violin(axes[0], "ext_trP_infantry",  "Infantry",  "tr(P)", dfs_main)
    _violin(axes[1], "ext_trP_tank",      "Tank",      "tr(P)", dfs_main)
    _violin(axes[2], "ext_trP_artillery", "Artillery", "tr(P)", dfs_main)
    _save(fig, out_dir, "fig_trP_type.png")

    # ── 4. Safety / Coordination ───────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Safety & Coordination Metrics  [4×20, N=100 ep]\n"
                 "drone_prox_events = Inter-UAV proximity (safety distance violation)",
                 fontweight="bold")
    _violin(axes[0], "drone_prox_events", "Inter-UAV Proximity Events  ↓\n(Safety)", "count",      dfs_main)
    _violin(axes[1], "active_at_end",     "Active Drones at End  ↑\n(Coordination)", "count",      dfs_main)
    _violin(axes[2], "mean_battery",      "Battery Remaining  ↑",                    "normalized", dfs_main)
    _violin(axes[3], "rtb_count",         "RTB Count  ↓",                            "count",      dfs_main)
    _save(fig, out_dir, "fig_safety.png")

    # ── 5. Scale 확장성 ────────────────────────────────────────────────
    scale_list = [s for s in scales if s in all_dfs]
    if len(scale_list) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Zero-shot Scale Generalization  (4→8→12 drones)",
                     fontweight="bold", fontsize=13)
        ax_f = axes.flatten()

        scale_cols = {
            "4x20":  (4,  20),
            "8x40":  (8,  40),
            "12x60": (12, 60),
        }
        scale_metrics = [
            ("unc_reduction_pct",        "Uncertainty Reduction %  ↑", "%"),
            ("ext_auc_mean_trP",         "AUC mean tr(P)  ↓",          "tr(P)"),
            ("committed_ratio",          "Committed ratio  ↑",          "ratio"),
            ("drone_prox_events",        "Inter-UAV Proximity Events  ↓","count"),
        ]
        x_labels = [s for s in scale_list]
        x_pos    = np.arange(len(x_labels))

        for ax, (col, title, ylabel) in zip(ax_f, scale_metrics):
            for m in MODELS:
                means = []; stds = []
                for scale in scale_list:
                    arr = _get(all_dfs[scale].get(m["key"]), col)
                    means.append(arr.mean() if len(arr) else np.nan)
                    stds.append(arr.std()  if len(arr) else 0.0)
                means = np.array(means); stds = np.array(stds)
                ax.plot(x_pos, means, marker=m["marker"], color=m["color"],
                        label=m["label"], linewidth=2.0, linestyle=m["ls"])
                ax.fill_between(x_pos, means-stds, means+stds,
                                color=m["color"], alpha=0.12)
            ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")

        _save(fig, out_dir, "fig_scale.png")
    else:
        print("  [스킵] scale 수 부족 — fig_scale.png 생략")

    # ── 6. Summary bar (핵심 지표 한눈에) ─────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Summary — 5 Methods  [4×20, N=100 ep, mean ± std]",
                 fontweight="bold", fontsize=13)
    summary = [
        ("unc_reduction_pct",        "Uncertainty\nReduction %  ↑",  "%"),
        ("ext_auc_mean_trP",         "AUC\nmean tr(P)  ↓",           "tr(P)"),
        ("committed_ratio",          "Committed\nratio  ↑",           "ratio"),
        ("stale_ratio_norm",         "Stale\nratio  ↓",              "ratio"),
        ("ext_time_under_good_track","Time under\ngood track  ↑",     "ratio"),
        ("drone_prox_events",        "Inter-UAV\nProximity  ↓",      "count"),
        ("active_at_end",            "Active\nat end  ↑",            "count"),
        ("mean_battery",             "Battery\nremaining  ↑",        "normalized"),
    ]
    for ax, (col, title, ylabel) in zip(axes.flatten(), summary):
        _bar(ax, col, title, ylabel, dfs_main)
    _save(fig, out_dir, "fig_summary.png")

    # ── 7. Scatter — uncertainty vs tr(P) ─────────────────────────────
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in MODELS:
        df  = dfs_main.get(m["key"])
        unc = _get(df, "uncertainty_pct")
        trp = _get(df, "ext_mean_trP_final")
        if len(unc) == 0 or len(trp) == 0: continue
        n = min(len(unc), len(trp))
        ax.scatter(unc[:n], trp[:n], alpha=0.35, color=m["color"],
                   s=35, label=m["label"], marker=m["marker"])
        ax.scatter([unc[:n].mean()], [trp[:n].mean()],
                   color=m["color"], s=250, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_xlabel("Final Uncertainty (%)  ↓", fontsize=11)
    ax.set_ylabel("Mean tr(P) final  ↓", fontsize=11)
    ax.set_title("Search-Track Trade-off  [4×20]  (↙ better)\n★ = mean per method",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_scatter.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [저장] fig_scatter.png")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v13 Scale Experiment 결과 분석")
    parser.add_argument("--no_plot", action="store_true", help="그래프 생략")
    parser.add_argument("--scales", nargs="+",
                        choices=["4x20", "8x40", "12x60"],
                        default=["4x20", "8x40", "12x60"])
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  analyze_results_v13.py  —  5 method × {len(args.scales)} scale")
    print(f"  입력: {IN_DIR}")
    print(f"  출력: {OUT_DIR}")
    print(f"{'='*70}")

    # ── 데이터 로드 ────────────────────────────────────────────────────
    print(f"\n[데이터 로드]")
    all_dfs = load_all_scales(args.scales)

    if not any(all_dfs.values()):
        print("[오류] 로드된 CSV 없음. eval_v13_scale_all.py 먼저 실행하세요.")
        return

    # ── scale별 비교표 ─────────────────────────────────────────────────
    print(f"\n[비교표 생성]")
    all_tables = []
    for scale in args.scales:
        dfs = all_dfs.get(scale, {})
        if not dfs: continue
        table = make_comparison_table(dfs, scale)
        all_tables.append(table)
        path = os.path.join(OUT_DIR, f"comparison_table_{scale}.csv")
        table.to_csv(path, index=False)
        print(f"  → {path}")

    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "comparison_table_all.csv"), index=False)
        print(f"  → comparison_table_all.csv (전체 합산)")

    # ── 콘솔 요약 ──────────────────────────────────────────────────────
    print_summary(all_dfs, args.scales)

    # ── 그래프 ────────────────────────────────────────────────────────
    if not args.no_plot:
        print(f"\n[그래프 생성]")
        try:
            make_plots(all_dfs, args.scales, OUT_DIR)
        except ImportError as e:
            print(f"  matplotlib 없음 → pip install matplotlib  ({e})")

    print(f"\n{'='*70}")
    print(f"  완료  →  {OUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
