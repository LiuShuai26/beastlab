#!/usr/bin/env python3
"""Test an exported ONNX policy in the Beast environment.

Usage:
    python scripts/test_onnx.py --model beast_ppo.onnx --module_name HumanoidEnv --episodes 10
"""

import argparse
import numpy as np
import onnxruntime as ort
from beastlab.env_loader import make_beast_gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--module_name", type=str, default="HumanoidEnv")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model)
    env = make_beast_gym(args.module_name)

    ep_rewards = []
    ep_lengths = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            actions = sess.run(None, {"obs": obs[np.newaxis].astype(np.float32)})[0][0]
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += float(reward[0]) if hasattr(reward, '__len__') else float(reward)
            steps += 1

            if bool(terminated[0] if hasattr(terminated, '__len__') else terminated) or \
               bool(truncated[0] if hasattr(truncated, '__len__') else truncated):
                break

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        print(f"Episode {ep+1:3d}: reward={total_reward:8.2f}  length={steps}")

    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"  Reward: mean={np.mean(ep_rewards):.2f}  std={np.std(ep_rewards):.2f}  "
          f"min={np.min(ep_rewards):.2f}  max={np.max(ep_rewards):.2f}")
    print(f"  Length: mean={np.mean(ep_lengths):.1f}  std={np.std(ep_lengths):.1f}  "
          f"min={np.min(ep_lengths)}  max={np.max(ep_lengths)}")


if __name__ == "__main__":
    main()
