from __future__ import annotations

import os
import sys
import time
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# repository root — 논문 공개용 repo에서 개인 PC 절대경로 대신 사용
REPO_ROOT = Path(__file__).resolve().parent

MAX_STEPS_DEFAULT = 1200

SCALE_CONFIGS = {
    "4x20":  {"num_drones": 4,  "num_targets": 20, "k_active": 12},
    "8x40":  {"num_drones": 8,  "num_targets": 40, "k_active": 12},
    "12x60": {"num_drones": 12, "num_targets": 60, "k_active": 12},
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=str, default="4x20",
                   choices=list(SCALE_CONFIGS.keys()))
    p.add_argument("--n_steps", type=int, default=1000,
                   help="timed steps after warm-up")
    p.add_argument("--warmup", type=int, default=100,
                   help="warm-up steps before timing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_graph_token", type=str,
                   default=str(REPO_ROOT / "graph_token_mappo_v13" / "stage1"
                               / "graph_token_ep03000.pt"))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_if_cuda(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


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


def _drone_positions(env):
    base_env = getattr(env, "env", env)
    return np.array(
        [base_env.drone_positions[i][:2] for i in range(base_env.num_drones)],
        dtype=np.float64,
    )


def _target_positions(env):
    base_env = getattr(env, "env", env)
    return np.array(
        [base_env.target_positions[i][:3] for i in range(base_env.num_targets)],
        dtype=np.float64,
    )


def _run_ext_tracker_step(tracker, env):
    """
    Standalone TargetBeliefTracker update.
    이 값은 IMM-KF belief update overhead를 따로 보기 위한 참고 측정값이다.
    """
    import isaac_env_v13

    R_TRACK = isaac_env_v13.R_TRACK_M
    base_env = getattr(env, "env", env)

    if hasattr(tracker, "predict_all"):
        tracker.predict_all()

    drone_xy = _drone_positions(base_env)
    tgt_pos = _target_positions(base_env)
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

    tracker.update_target_states()


def mean_ms(times):
    return float(np.mean(times)) if times else 0.0


def std_ms(times):
    return float(np.std(times)) if times else 0.0


def p95_ms(times):
    return float(np.percentile(times, 95)) if times else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = SCALE_CONFIGS[args.scale]
    num_drones = cfg["num_drones"]
    num_targets = cfg["num_targets"]
    k_active = cfg["k_active"]

    print("\n" + "=" * 78)
    print("Graph-Token MAPPO Pipeline Runtime Breakdown")
    print(f"  Scale   : {args.scale} ({num_drones} UAVs / {num_targets} targets)")
    print(f"  Warm-up : {args.warmup} steps")
    print(f"  Timed   : {args.n_steps} steps")
    print(f"  Device  : {args.device}")
    print(f"  Ckpt    : {args.ckpt_graph_token}")
    print("=" * 78)

    from graph_token_mappo_v13 import (
        TrainConfig,
        MAPPOAgent,
        GraphTokenObsWrapper,
    )
    from isaac_env_v13 import TargetBeliefTracker

    base_env, env_module = _make_env_v13(num_drones, num_targets, k_active)
    env = GraphTokenObsWrapper(base_env)

    target_types = tuple(
        int(t.value) if hasattr(t, "value") else int(t)
        for t in base_env.target_types
    )

    ext_tracker = TargetBeliefTracker(
        base_env.num_targets,
        target_types,
        _target_positions(base_env),
    )

    cfg_agent = TrainConfig(device=args.device)
    agent = MAPPOAgent(cfg_agent)

    if os.path.isfile(args.ckpt_graph_token):
        agent.load(args.ckpt_graph_token)
        print(f"  checkpoint loaded: {args.ckpt_graph_token}")
    else:
        print(f"  [warning] checkpoint not found. Random weights used: {args.ckpt_graph_token}")

    agent.actor.eval()
    agent.critic.eval()
    agent.ep_prog = 1.0

    obs, gobs = env.reset()
    ext_tracker.reset(_target_positions(base_env))

    policy_times = []
    env_step_times = []
    ext_tracker_times = []
    full_loop_times = []

    total_steps = args.warmup + args.n_steps

    pbar = tqdm(range(total_steps), desc="pipeline timing", ncols=100, unit="step")

    for step in pbar:
        timed = step >= args.warmup

        # ------------------------------------------------------------
        # 1) Policy action selection
        # ------------------------------------------------------------
        sync_if_cuda(args.device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            acts, _, _, _, _ = agent.get_action_and_value(
                obs,
                gobs,
                deterministic=False,
            )

        sync_if_cuda(args.device)
        t1 = time.perf_counter()

        actions_np = acts.cpu().numpy()

        # ------------------------------------------------------------
        # 2) Environment step + wrapper observation generation
        # ------------------------------------------------------------
        t2 = time.perf_counter()
        (nobs, ngobs), rews, dones, info = env.step(actions_np)
        t3 = time.perf_counter()

        # ------------------------------------------------------------
        # 3) Standalone external tracker update
        # ------------------------------------------------------------
        t4 = time.perf_counter()
        _run_ext_tracker_step(ext_tracker, env)
        t5 = time.perf_counter()

        if timed:
            policy_times.append((t1 - t0) * 1000.0)
            env_step_times.append((t3 - t2) * 1000.0)
            ext_tracker_times.append((t5 - t4) * 1000.0)
            full_loop_times.append(((t1 - t0) + (t3 - t2)) * 1000.0)

        obs, gobs = nobs, ngobs

        if dones.all():
            obs, gobs = env.reset()
            ext_tracker.reset(_target_positions(base_env))

        if timed and (step - args.warmup + 1) % 100 == 0:
            pbar.set_postfix({
                "policy": f"{mean_ms(policy_times):.3f}ms",
                "env": f"{mean_ms(env_step_times):.3f}ms",
                "tracker": f"{mean_ms(ext_tracker_times):.3f}ms",
            })

    results = [
        {
            "component": "Policy action selection",
            "mean": mean_ms(policy_times),
            "std": std_ms(policy_times),
            "p95": p95_ms(policy_times),
        },
        {
            "component": "Environment step + obs.",
            "mean": mean_ms(env_step_times),
            "std": std_ms(env_step_times),
            "p95": p95_ms(env_step_times),
        },
        {
            "component": "Standalone IMM-KF tracker update",
            "mean": mean_ms(ext_tracker_times),
            "std": std_ms(ext_tracker_times),
            "p95": p95_ms(ext_tracker_times),
        },
        {
            "component": "Policy + environment loop",
            "mean": mean_ms(full_loop_times),
            "std": std_ms(full_loop_times),
            "p95": p95_ms(full_loop_times),
        },
    ]

    print("\n" + "=" * 78)
    print("Pipeline Runtime Breakdown")
    print("=" * 78)
    print(f"{'Component':<38} {'Mean ms':>10} {'Std ms':>10} {'P95 ms':>10}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['component']:<38} "
            f"{r['mean']:>10.3f} "
            f"{r['std']:>10.3f} "
            f"{r['p95']:>10.3f}"
        )
    print("=" * 78)

    print("\n% LaTeX table")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\resizebox{\linewidth}{!}{%")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Component & Mean (ms/step) & Std. & P95 \\")
    print(r"\midrule")
    for r in results:
        print(
            f"{r['component']} & "
            f"{r['mean']:.3f} & "
            f"{r['std']:.3f} & "
            f"{r['p95']:.3f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(
        r"\caption{Pipeline runtime breakdown of the proposed Graph-Token MAPPO "
        r"framework. The policy-action and environment-step measurements are taken "
        r"from actual evaluation rollouts. The standalone IMM-KF tracker update is "
        r"reported separately to estimate the cost of target-belief maintenance.}"
    )
    print(r"\label{tab:pipeline_runtime_breakdown}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
