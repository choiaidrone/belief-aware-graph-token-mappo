# run_seed_training_all.py
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable

REPO_ROOT = Path(__file__).resolve().parent
RUN_ROOT = REPO_ROOT / "seed_robustness_runs"

JOBS = [
    {
        "name": "graph_token",
        "script": REPO_ROOT / "graph_token_mappo_v13.py",
        "save_dir": RUN_ROOT / "graph_token",
        "final_ckpt_name": "graph_token_ep01500.pt",
    },
    {
        "name": "gat_mappo",
        "script": REPO_ROOT / "gat_mappo_v13.py",
        "save_dir": RUN_ROOT / "gat_mappo",
        "final_ckpt_name": "gat_paper_ep01500.pt",
    },
]

SEEDS = [0, 1, 2, 3, 4]


def run_one(job, seed):
    save_dir = job["save_dir"]

    # 각 학습 코드 내부에서 save_dir/stage1/seed{seed} 구조로 저장됨
    final_ckpt = save_dir / "stage1" / f"seed{seed}" / job["final_ckpt_name"]

    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['name']}_seed{seed}.log"

    if final_ckpt.exists():
        print(f"[SKIP] {job['name']} seed={seed} already complete")
        print(f"       {final_ckpt}")
        return

    cmd = [
        PYTHON,
        str(job["script"]),
        "--stage", "1",
        "--seed", str(seed),
        "--save_dir", str(save_dir),
        "--episodes", "1500",
    ]

    print("=" * 80)
    print(f"[RUN] {job['name']} seed={seed}")
    print(" ".join(cmd))
    print(f"[SAVE ROOT] {save_dir}")
    print(f"[LOG] {log_path}")
    print("=" * 80)

    start = time.time()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        ret = proc.wait()

    elapsed = time.time() - start

    if ret != 0:
        raise RuntimeError(
            f"{job['name']} seed={seed} failed with return code {ret}. "
            f"Check log: {log_path}"
        )

    print(f"[DONE] {job['name']} seed={seed} elapsed={elapsed/3600:.2f} h")


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[SEED ROBUSTNESS TRAINING]")
    print(f"Run root: {RUN_ROOT}")
    print(f"Seeds   : {SEEDS}")
    print("=" * 80)

    for seed in SEEDS:
        for job in JOBS:
            run_one(job, seed)

    print("\nAll seed trainings completed.")
    print(f"Results saved under: {RUN_ROOT}")


if __name__ == "__main__":
    main()
