#!/usr/bin/env python3
"""Train a PPO agent using Stable Baselines3 with a headless Beast .so env.

Usage:
    python scripts/train_sb3_headless.py HumanoidEnv
    python scripts/train_sb3_headless.py HumanoidEnv --timesteps 5000000
    python scripts/train_sb3_headless.py HumanoidEnv --project path/to/Project.project
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from beastlab import make_beast_gym


def main():
    parser = argparse.ArgumentParser(description="Train PPO on a headless Beast env")
    parser.add_argument("module", help="Module name (e.g. HumanoidEnv)")
    parser.add_argument("--project", default=None, help="Path to .project file")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--save", default="beast_ppo", help="Path to save the trained model")
    args = parser.parse_args()

    env = make_beast_gym(args.module, project_path=args.project)
    eval_env = make_beast_gym(args.module, project_path=args.project)

    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,
        batch_size=1024,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 128],
            log_std_init=-1.0,
        ),
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.save)
    print(f"Model saved to {args.save}.zip")


if __name__ == "__main__":
    main()
