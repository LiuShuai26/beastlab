#!/usr/bin/env python3
"""Example: Train a PPO agent using Stable Baselines3 with the Beast editor.

Prerequisites:
    pip install beastlab[train]

Usage:
    1. Open your scene in the Beast editor
    2. Click "Training" in the Habitat panel to start the MLServer
    3. Run this script:

        python train_sb3.py --port 5555 --timesteps 100000
"""

import argparse

from stable_baselines3 import PPO

from beastlab import BeastClientEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO on a Beast environment")
    parser.add_argument(
        "--host", default="127.0.0.1", help="MLServer host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="MLServer port (default: 5555)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--save", default="beast_ppo", help="Path to save the trained model"
    )
    args = parser.parse_args()

    env = BeastClientEnv(host=args.host, port=args.port)
    print(f"Connected: obs={env.observation_size}, act={env.continuous_action_size}")

    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,  # 每次 rollout 收集步数（单环境要大）
        batch_size=1024,  # mini-batch
        n_epochs=10,  # 每次 rollout 训练轮数（单环境数据少，多复用）
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # 小一点的熵，鼓励探索但不要太随机
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 128],  # 2D 12-DOF 不需要太大网络
            log_std_init=-1.0,  # 初始动作标准差 ≈ 0.37，减少开局乱甩
        ),
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"Model saved to {args.save}.zip")

    env.close()


if __name__ == "__main__":
    main()
