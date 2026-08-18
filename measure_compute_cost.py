"""
measure_compute_cost.py
=======================
5개 방법 계산 비용 측정 스크립트.
  1. Random Walk
  2. SSM (SSMHeuristic)
  3. CC-MPSO (CCMPSOHeuristic)
  4. GAT-MAPPO
  5. Graph-Token MAPPO (Proposed)

측정 항목:
  - Trainable parameters (Actor / Critic / Total)
  - Decision time (ms/step): warmup 후 n_repeat 번 평균

실행:
    python measure_compute_cost.py
    python measure_compute_cost.py --n_drones 4 --n_repeat 1000 --warmup 100
    python measure_compute_cost.py --gat_ckpt gat_paper/stage1/gat_paper_ep03000.pt

수정 이력:
  v2: measure_time() warmup/timed 루프 분리
      SSM/CC-MPSO env_pool 미리 생성 (timing 밖)
      GAT/Graph-Token actor-only 측정으로 변경
      LaTeX caption 에 CPU/GPU 구분 명시
"""

from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
from tqdm import tqdm

# ── path ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ════════════════════════════════════════════════════════════════════════
# Args
# ════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_drones",  type=int, default=4)
    p.add_argument("--n_repeat",  type=int, default=1000,
                   help="timed 반복 횟수 (warmup 제외)")
    p.add_argument("--warmup",    type=int, default=100,
                   help="GPU warm-up 횟수 (시간 측정 제외)")
    p.add_argument("--gat_ckpt",  type=str,
                   default=r"gat_paper/stage1/gat_paper_ep03000.pt")
    p.add_argument("--gt_ckpt",   type=str,
                   default=r"graph_token_mappo_v13/stage1/graph_token_ep03000.pt")
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# Mock env for heuristics
# ════════════════════════════════════════════════════════════════════════
class MockEnv:
    """SSM / CC-MPSO 에 필요한 최소 env 인터페이스."""
    def __init__(self, n_drones: int):
        self.num_drones            = n_drones
        self.drone_active          = np.ones(n_drones, dtype=bool)
        self.drone_battery         = np.ones(n_drones, dtype=np.float32)
        self.drone_positions       = np.random.uniform(
            low=[-20, -50, 0], high=[80, 50, 10],
            size=(n_drones, 3)).astype(np.float32)
        n_targets                  = 2
        self.target_positions      = np.random.uniform(
            low=[-20, -50, 0], high=[80, 50, 5],
            size=(n_targets, 3)).astype(np.float32)
        self.target_is_camouflaged = np.zeros(n_targets, dtype=bool)
        self.detection_events      = []

        class FakeDS:
            def __init__(self):
                self.U = np.random.rand(100, 100).astype(np.float32)
            def _pool(self, arr):
                return arr.reshape(10, 10, 10, 10).mean(axis=(1, 3))
        self.ds_map = FakeDS()


# ════════════════════════════════════════════════════════════════════════
# 측정 유틸
# ════════════════════════════════════════════════════════════════════════
def measure_time(fn, n_repeat: int, warmup: int, use_cuda: bool,
                 desc: str = "") -> float:
    """
    warm-up을 먼저 warmup번 수행한 뒤,
    별도로 n_repeat번 timed run을 수행하여 평균 ms 반환.
    → n_repeat 값 그대로 논문에 표기 가능.
    """
    # ── warm-up (측정 제외) ──────────────────────────────────────────────
    for _ in tqdm(range(warmup), desc=f"  {desc} warm-up",
                  ncols=70, leave=False, unit="call"):
        if use_cuda:
            torch.cuda.synchronize()
        fn()
        if use_cuda:
            torch.cuda.synchronize()

    # ── timed run ───────────────────────────────────────────────────────
    times = []
    for _ in tqdm(range(n_repeat), desc=f"  {desc} timing",
                  ncols=70, leave=False, unit="call"):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)   # ms

    return float(np.mean(times))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ════════════════════════════════════════════════════════════════════════
# 1. Random Walk
# ════════════════════════════════════════════════════════════════════════
def bench_random_walk(n_drones: int, n_repeat: int, warmup: int) -> dict:
    print("\n[1/5] Random Walk ...")

    def fn():
        return np.random.randint(0, 4, size=n_drones)

    t = measure_time(fn, n_repeat, warmup, use_cuda=False, desc="Random Walk")
    return {
        "method"       : "Random Walk",
        "type"         : "Rule-free",
        "actor_params" : 0,
        "critic_params": 0,
        "total_params" : 0,
        "time_ms"      : t,
        "device_tag"   : "CPU",
        "online_opt"   : "No",
    }


# ════════════════════════════════════════════════════════════════════════
# 2. SSM
# ════════════════════════════════════════════════════════════════════════
def bench_ssm(n_drones: int, n_repeat: int, warmup: int) -> dict:
    print("[2/5] SSM ...")
    from heuristics_v13 import SSMHeuristic, SSMConfig

    total_calls = warmup + n_repeat
    # env_pool 미리 생성 → timing loop 안에서 입력 생성 시간 제외
    env_pool = [MockEnv(n_drones) for _ in range(total_calls)]
    agent = SSMHeuristic(env=env_pool[0], config=SSMConfig())
    idx = 0

    def fn():
        nonlocal idx
        env = env_pool[idx % total_calls]
        idx += 1
        return agent.compute_actions(env=env)

    t = measure_time(fn, n_repeat, warmup, use_cuda=False, desc="SSM")
    return {
        "method"       : "SSM",
        "type"         : "Rule-based",
        "actor_params" : 0,
        "critic_params": 0,
        "total_params" : 0,
        "time_ms"      : t,
        "device_tag"   : "CPU",
        "online_opt"   : "No",
    }


# ════════════════════════════════════════════════════════════════════════
# 3. CC-MPSO
# ════════════════════════════════════════════════════════════════════════
def bench_ccmpso(n_drones: int, n_repeat: int, warmup: int) -> dict:
    print("[3/5] CC-MPSO ...")
    from heuristics_v13 import CCMPSOHeuristic, CCMPSOConfig

    total_calls = warmup + n_repeat
    env_pool = [MockEnv(n_drones) for _ in range(total_calls)]
    agent = CCMPSOHeuristic(env=env_pool[0], config=CCMPSOConfig())
    idx = 0

    def fn():
        nonlocal idx
        env = env_pool[idx % total_calls]
        idx += 1
        return agent.compute_actions(env=env)

    t = measure_time(fn, n_repeat, warmup, use_cuda=False, desc="CC-MPSO")
    return {
        "method"       : "CC-MPSO",
        "type"         : "Optimization-based",
        "actor_params" : 0,
        "critic_params": 0,
        "total_params" : 0,
        "time_ms"      : t,
        "device_tag"   : "CPU",
        "online_opt"   : "Yes",
    }


# ════════════════════════════════════════════════════════════════════════
# 4. GAT-MAPPO  —  actor-only 측정
# ════════════════════════════════════════════════════════════════════════
def bench_gat_mappo(n_drones: int, n_repeat: int, warmup: int,
                    ckpt_path: str, device: str) -> dict:
    print("[4/5] GAT-MAPPO ...")
    from gat_mappo_v13 import GATMAPPOAgent, TrainConfig, GAT_OBS_DIM

    cfg   = TrainConfig(device=device)
    agent = GATMAPPOAgent(cfg, n_drones)

    ckpt_full = os.path.join(ROOT, ckpt_path)
    if os.path.isfile(ckpt_full):
        agent.load(ckpt_full)
        print(f"  checkpoint: {ckpt_full}")
    else:
        print(f"  ⚠ checkpoint 없음, random weights 사용: {ckpt_full}")

    agent.actor.eval()
    agent.critic.eval()
    dev      = torch.device(device)
    use_cuda = device.startswith("cuda")

    # obs pool 미리 생성 (warmup + n_repeat)
    total_calls = warmup + n_repeat
    obs_pool = [
        torch.from_numpy(
            np.random.randn(n_drones, GAT_OBS_DIM).astype(np.float32)
        ).unsqueeze(0).to(dev)          # (1, N, D)
        for _ in range(total_calls)
    ]
    idx = 0

    # actor-only: logits (1, N, A) → argmax per drone
    def fn():
        nonlocal idx
        obs_t = obs_pool[idx % total_calls]
        idx  += 1
        with torch.inference_mode():
            logits = agent.actor(obs_t)          # (1, N, ACTION_DIM)
            _      = logits.argmax(-1).squeeze(0)  # (N,)

    t = measure_time(fn, n_repeat, warmup, use_cuda, desc="GAT-MAPPO")

    ap = count_params(agent.actor)
    cp = count_params(agent.critic)
    return {
        "method"       : "GAT-MAPPO",
        "type"         : "Learning-based",
        "actor_params" : ap,
        "critic_params": cp,
        "total_params" : ap + cp,
        "time_ms"      : t,
        "device_tag"   : device.upper(),
        "online_opt"   : "No",
    }


# ════════════════════════════════════════════════════════════════════════
# 5. Graph-Token MAPPO (Proposed)  —  actor-only 측정
# ════════════════════════════════════════════════════════════════════════
def bench_graph_token(n_drones: int, n_repeat: int, warmup: int,
                      ckpt_path: str, device: str) -> dict:
    print("[5/5] Graph-Token MAPPO (Proposed) ...")
    from graph_token_mappo_v13 import MAPPOAgent, TrainConfig, OBS_DIM, GAT_OBS_DIM

    cfg   = TrainConfig(device=device)
    agent = MAPPOAgent(cfg)

    ckpt_full = os.path.join(ROOT, ckpt_path)
    if os.path.isfile(ckpt_full):
        agent.load(ckpt_full)
        print(f"  checkpoint: {ckpt_full}")
    else:
        print(f"  ⚠ checkpoint 없음, random weights 사용: {ckpt_full}")

    agent.actor.eval()
    agent.critic.eval()
    dev      = torch.device(device)
    use_cuda = device.startswith("cuda")

    total_calls = warmup + n_repeat
    obs_pool = [
        (
            torch.from_numpy(
                np.random.randn(n_drones, OBS_DIM    ).astype(np.float32)
            ).unsqueeze(0).to(dev),     # (1, N, OBS_DIM)
            torch.from_numpy(
                np.random.randn(n_drones, GAT_OBS_DIM).astype(np.float32)
            ).unsqueeze(0).to(dev),     # (1, N, GAT_OBS_DIM)
        )
        for _ in range(total_calls)
    ]
    idx = 0

    # actor-only: ep_prog=1.0 (fully trained) 고정
    def fn():
        nonlocal idx
        obs_t, gobs_t = obs_pool[idx % total_calls]
        idx += 1
        with torch.inference_mode():
            logits, _ = agent.actor(obs_t, gobs_t, ep_prog=1.0)
            _         = logits.argmax(-1).squeeze(0)   # (N,)

    t = measure_time(fn, n_repeat, warmup, use_cuda, desc="Graph-Token")

    ap = count_params(agent.actor)
    cp = count_params(agent.critic)
    return {
        "method"       : "Graph-Token MAPPO",
        "type"         : "Learning-based",
        "actor_params" : ap,
        "critic_params": cp,
        "total_params" : ap + cp,
        "time_ms"      : t,
        "device_tag"   : device.upper(),
        "online_opt"   : "No",
    }


# ════════════════════════════════════════════════════════════════════════
# 결과 출력
# ════════════════════════════════════════════════════════════════════════
def fmt_params(n: int) -> str:
    if n == 0:   return "--"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)


def print_results(results: list, n_repeat: int, warmup: int):
    print("\n" + "="*88)
    print("Computational Cost Analysis")
    print(f"  Timed runs: {n_repeat}  |  Warm-up: {warmup}")
    print("="*88)
    header = (f"{'Method':<26} {'Type':<22} {'Actor':>9} {'Critic':>9}"
              f" {'Total':>9} {'Device':>6} {'ms/step':>9}")
    print(header)
    print("-"*88)
    for r in results:
        print(f"{r['method']:<26} {r['type']:<22} "
              f"{fmt_params(r['actor_params']):>9} "
              f"{fmt_params(r['critic_params']):>9} "
              f"{fmt_params(r['total_params']):>9} "
              f"{r['device_tag']:>6} "
              f"{r['time_ms']:>8.3f}ms")
    print("="*88)


def print_latex(results: list, gpu_name: str, n_repeat: int, warmup: int):
    print("\n% ── LaTeX Table ──────────────────────────────────────────────")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\resizebox{\linewidth}{!}{")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Method & Type & Trainable Params & Decision Time (ms/step)"
          r" & Device & Online Opt. \\")
    print(r"\midrule")
    for r in results:
        tp = fmt_params(r['total_params'])
        t  = f"{r['time_ms']:.3f}" if r['time_ms'] > 0 else "--"
        print(f"{r['method']} & {r['type']} & {tp} & {t}"
              f" & {r['device_tag']} & {r['online_opt']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    caption = (
        r"\caption{Computational cost comparison of all methods. "
        r"Trainable parameters are reported for learning-based methods only. "
        r"Decision time is the average actor-only forward-pass latency "
        rf"over {n_repeat:,} timed runs after {warmup} warm-up calls. "
        r"Rule-based and optimization-based methods are executed on CPU; "
        r"learning-based methods are evaluated on a single "
        + gpu_name +
        r". CC-MPSO performs online iterative optimization at every decision step.}"
    )
    print(caption)
    print(r"\label{tab:computational_cost}")
    print(r"\end{table}")


# ════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════
def main():
    args     = parse_args()
    N        = args.n_drones
    device   = args.device
    use_cuda = device.startswith("cuda")

    print("=== Computational Cost Measurement ===")
    print(f"  n_drones : {N}")
    print(f"  device   : {device}")
    print(f"  n_repeat : {args.n_repeat}  warmup={args.warmup}")

    if use_cuda:
        props    = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        print(f"  GPU      : {gpu_name}")
    else:
        import platform
        gpu_name = f"CPU ({platform.processor()})"

    methods = [
        ("Random Walk",       lambda: bench_random_walk(N, args.n_repeat, args.warmup)),
        ("SSM",               lambda: bench_ssm        (N, args.n_repeat, args.warmup)),
        ("CC-MPSO",           lambda: bench_ccmpso     (N, args.n_repeat, args.warmup)),
        ("GAT-MAPPO",         lambda: bench_gat_mappo  (N, args.n_repeat, args.warmup,
                                                        args.gat_ckpt, device)),
        ("Graph-Token MAPPO", lambda: bench_graph_token(N, args.n_repeat, args.warmup,
                                                        args.gt_ckpt,  device)),
    ]

    results = []
    overall = tqdm(methods, desc="Overall progress", ncols=70, unit="method")
    for name, bench_fn in overall:
        overall.set_postfix(method=name)
        results.append(bench_fn())

    print_results(results, args.n_repeat, args.warmup)
    print_latex(results, gpu_name, args.n_repeat, args.warmup)


if __name__ == "__main__":
    main()
