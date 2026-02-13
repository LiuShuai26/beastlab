"""Sample Factory integration for Beast headless training."""

from pathlib import Path

from sample_factory.envs.env_utils import register_env

from beastlab.env_loader import load_beast_env, BeastGymWrapper

_ROOT = Path(__file__).parent.parent


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
