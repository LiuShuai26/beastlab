"""Dynamic loader for Beast GameEnv .so modules from the envs/ directory."""

import importlib.util
from pathlib import Path

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
