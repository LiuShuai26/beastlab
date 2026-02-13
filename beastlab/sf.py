"""Sample Factory integration for Beast headless training."""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sample_factory.envs.env_utils import register_env

from beastlab.env_loader import load_beast_env

_ROOT = Path(__file__).parent.parent


class BeastGymWrapper(gym.Env):
    """Minimal Gymnasium wrapper so Sample Factory recognises the env."""

    def __init__(self, raw_env):
        super().__init__()
        self._env = raw_env
        self._env.reset()

        obs_size = self._env.observation_size()
        action_info = self._env.action_space()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32,
        )
        cont_size = action_info["continuous_size"]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(cont_size,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, info = self._env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, rewards, terminated, truncated, info = self._env.step(action)
        return (
            np.asarray(obs, dtype=np.float32),
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )


def make_beast_env(full_env_name, cfg, env_config, render_mode=None):
    """Factory function called by Sample Factory to create environment instances."""
    module = load_beast_env(cfg.module_name)
    EnvClass = getattr(module, cfg.module_name)
    project_path = getattr(cfg, "project_path", None)
    if project_path:
        project_path = Path(project_path)
        if not project_path.is_absolute():
            project_path = _ROOT / project_path
        raw_env = EnvClass(str(project_path))
    else:
        raw_env = EnvClass()
    return BeastGymWrapper(raw_env)


def register_beast_envs():
    """Register Beast environments with Sample Factory."""
    register_env("beast", make_beast_env)


def add_beast_args(parser):
    """Add Beast-specific CLI arguments to Sample Factory's argument parser."""
    parser.add_argument(
        "--module_name", type=str, required=True,
        help="pybind11 module name (e.g., HumanoidEnv)",
    )
    parser.add_argument(
        "--project_path", type=str, default=None,
        help="Path to .project file (optional if scene is baked into .so)",
    )
