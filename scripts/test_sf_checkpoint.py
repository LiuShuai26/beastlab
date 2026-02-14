#!/usr/bin/env python3
"""Test a Sample Factory checkpoint in the Beast environment.

Usage:
    python scripts/test_sf_checkpoint.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --episodes 10

    # Test best checkpoint
    python scripts/test_sf_checkpoint.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --episodes 10 --load_checkpoint_kind best
"""

import sys
import numpy as np
import torch
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg, load_from_checkpoint
from sample_factory.enjoy import load_state_dict, make_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.normalize import prepare_and_normalize_obs
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.utils.utils import preprocess_actions

from beastlab.sf import register_beast_envs, add_beast_args


def main():
    register_beast_envs()

    parser, _ = parse_sf_args(evaluation=True)
    add_beast_args(parser)
    parser.add_argument("--episodes", type=int, default=10)
    cfg = parse_full_cfg(parser)
    cfg = load_from_checkpoint(cfg)

    env = make_env(cfg)
    env_info = extract_env_info(env, cfg)
    device = torch.device("cpu")

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)

    ep_rewards = []
    ep_lengths = []

    with torch.no_grad():
        for ep in range(cfg.episodes):
            obs, _ = env.reset()
            rnn_states = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
            total_reward = 0.0
            steps = 0

            while True:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_out = actor_critic(normalized_obs, rnn_states)
                actions = policy_out["actions"]
                if actions.ndim == 1:
                    actions = actions.unsqueeze(-1)
                actions = preprocess_actions(env_info, actions)
                rnn_states = policy_out["new_rnn_states"]

                obs, reward, terminated, truncated, _ = env.step(actions)

                total_reward += float(reward[0]) if hasattr(reward, '__len__') else float(reward)
                steps += 1

                done = (bool(terminated[0]) if hasattr(terminated, '__len__') else bool(terminated)) or \
                       (bool(truncated[0]) if hasattr(truncated, '__len__') else bool(truncated))
                if done:
                    break

            ep_rewards.append(total_reward)
            ep_lengths.append(steps)
            print(f"Episode {ep+1:3d}: reward={total_reward:8.2f}  length={steps}")

    print(f"\n--- Summary ({cfg.episodes} episodes) ---")
    print(f"  Reward: mean={np.mean(ep_rewards):.2f}  std={np.std(ep_rewards):.2f}  "
          f"min={np.min(ep_rewards):.2f}  max={np.max(ep_rewards):.2f}")
    print(f"  Length: mean={np.mean(ep_lengths):.1f}  std={np.std(ep_lengths):.1f}  "
          f"min={np.min(ep_lengths)}  max={np.max(ep_lengths)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
