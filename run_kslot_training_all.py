"""
run_kslot_training_all.py
==========================
Graph-Token MAPPO K-slot (active-track token budget) sensitivity 자동 학습.

K = 8, 12, 16 을 동일한 reduced training budget(1500 episodes, stage1),
seed 0 고정으로 순차 학습한다. (appendix diagnostic 용도)

전제:
  - graph_token_mappo_v13_k8.py / _k12.py / _k16.py 가 repository root에 이미 존재
  - 각 파일의 K_ACTIVE / OBS_DIM / OBS_BATTERY_START / DroneSwarmEnv(k_active=...)
    가 K에 맞게 세팅되어 있어야 함
  - 각 파일의 CurriculumConfig.stage_episodes[1] == TARGET_EP 이어야 함
    (다르면 checkpoint 파일명이 안 맞아 [SKIP] 판정이 틀어짐)

실행:
    cd /path/to/repository
    conda activate drone_env_310
    python run_kslot_training_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parent
RUN_ROOT = REPO_ROOT / "kslot_runs"

# K-slot sensitivity는 diagnostic이므로 reduced budget(1500)으로 고정.
# 반드시 graph_token_mappo_v13_k*.py 안의 stage_episodes[1] 값과 같아야 함.
TARGET_EP = 1500

JOBS = [
    {
        "name": "graph_token_k8",
        "k": 8,
        "script": REPO_ROOT / "graph_token_mappo_v13_k8.py",
        "save_dir": RUN_ROOT / "graph_token_k8",
        "final_ckpt_name": f"graph_token_ep{TARGET_EP:05d}.pt",
    },
    {
        "name": "graph_token_k12",
        "k": 12,
        "script": REPO_ROOT / "graph_token_mappo_v13_k12.py",
        "save_dir": RUN_ROOT / "graph_token_k12",
        "final_ckpt_name": f"graph_token_ep{TARGET_EP:05d}.pt",
    },
    {
        "name": "graph_token_k16",
        "k": 16,
        "script": REPO_ROOT / "graph_token_mappo_v13_k16.py",
        "save_dir": RUN_ROOT / "graph_token_k16",
        "final_ckpt_name": f"graph_token_ep{TARGET_EP:05d}.pt",
    },
]

SEED = 0


def run_one(job):
    save_dir = job["save_dir"]
    # 학습 코드 내부에서 save_dir/stage1/seed{seed} 구조로 저장됨
    final_ckpt = save_dir / "stage1" / f"seed{SEED}" / job["final_ckpt_name"]

    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['name']}_seed{SEED}.log"

    if final_ckpt.exists():
        print(f"[SKIP] {job['name']} (K={job['k']}) seed={SEED} already complete")
        print(f"       {final_ckpt}")
        return

    if not job["script"].exists():
        raise FileNotFoundError(f"Script not found: {job['script']}")

    cmd = [
        PYTHON,
        str(job["script"]),
        "--stage", "1",
        "--seed", str(SEED),
        "--save_dir", str(save_dir),
    ]

    print("=" * 80)
    print(f"[RUN] {job['name']} | K={job['k']} | seed={SEED} | target_ep={TARGET_EP}")
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
            f"{job['name']} seed={SEED} failed with return code {ret}. "
            f"Check log: {log_path}"
        )

    print(f"[DONE] {job['name']} seed={SEED} elapsed={elapsed/3600:.2f} h")


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[K-SLOT SENSITIVITY TRAINING]")
    print(f"Run root : {RUN_ROOT}")
    print(f"Seed     : {SEED}")
    print(f"Target ep: {TARGET_EP}  (reduced-budget diagnostic)")
    print("=" * 80)

    for job in JOBS:
        run_one(job)

    print("\nAll K-slot trainings completed.")
    print(f"Results saved under: {RUN_ROOT}")


if __name__ == "__main__":
    main()
