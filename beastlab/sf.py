"""Sample Factory integration for Beast headless training."""

from sample_factory.envs.env_utils import register_env

from beastlab.env_loader import load_beast_env, BeastGymWrapper


def make_beast_env(full_env_name, cfg, env_config, render_mode=None):
    """Factory function called by Sample Factory to create environment instances."""
    module = load_beast_env(cfg.module_name)
    EnvClass = getattr(module, cfg.module_name)
    return BeastGymWrapper(EnvClass())


def register_beast_envs():
    """Register Beast environments with Sample Factory."""
    register_env("beast", make_beast_env)


def add_beast_args(parser):
    """Add Beast-specific CLI arguments to Sample Factory's argument parser."""
    parser.add_argument(
        "--module_name", type=str, required=True,
        help="pybind11 module name (e.g., HumanoidEnv)",
    )
