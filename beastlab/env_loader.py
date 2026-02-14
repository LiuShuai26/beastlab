"""Dynamic loader for Beast GameEnv .so modules from the envs/ directory."""

import importlib.util
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ENVS_DIR = Path(__file__).parent.parent / "envs"


def load_beast_env(name: str):
    """Load a .so env module by name, e.g. 'HumanoidEnv'"""
    so_files = list(ENVS_DIR.glob(f"{name}.cpython-*.so"))
    if not so_files:
        raise FileNotFoundError(f"No .so found for '{name}' in {ENVS_DIR}")
    spec = importlib.util.spec_from_file_location(name, so_files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BeastGymWrapper(gym.Env):
    """Gymnasium wrapper around a raw Beast .so env."""

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
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, rewards, terminated, truncated, info = self._env.step(action)
        return (
            np.asarray(obs, dtype=np.float32),
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )


def make_beast_gym(module_name: str):
    """Load a .so env and return it wrapped as a Gymnasium env."""
    module = load_beast_env(module_name)
    EnvClass = getattr(module, module_name)
    return BeastGymWrapper(EnvClass())
