"""Benchmark single-env step throughput (FPS).

Usage:
    python scripts/bench_env.py HumanoidEnv
    python scripts/bench_env.py HumanoidEnv --steps 10000
"""

import argparse
import time

import numpy as np

from beastlab.env_loader import load_beast_env


def bench(module_name, num_steps=5000):
    module = load_beast_env(module_name)
    EnvClass = getattr(module, module_name)
    env = EnvClass()

    obs, info = env.reset()
    action_info = env.action_space()
    cont_size = action_info["continuous_size"]

    # warmup
    for _ in range(100):
        action = np.random.uniform(-1, 1, size=(cont_size,)).astype(np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)
        if bool(terminated[0]) or bool(truncated[0]):
            obs, info = env.reset()

    # benchmark
    obs, info = env.reset()
    start = time.perf_counter()
    for _ in range(num_steps):
        action = np.random.uniform(-1, 1, size=(cont_size,)).astype(np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)
        if bool(terminated[0]) or bool(truncated[0]):
            obs, info = env.reset()
    elapsed = time.perf_counter() - start

    fps = num_steps / elapsed
    print(f"{module_name}: {num_steps} steps in {elapsed:.2f}s = {fps:.0f} FPS")


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-env FPS")
    parser.add_argument("module", help="Module name (e.g. HumanoidEnv)")
    parser.add_argument("--steps", type=int, default=5000, help="Steps to benchmark (default: 5000)")
    args = parser.parse_args()
    bench(args.module, num_steps=args.steps)


if __name__ == "__main__":
    main()
