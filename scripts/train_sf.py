"""Train with Sample Factory (high-throughput async PPO).

Usage:
    # Train with config file (all params including env/module specified in yaml)
    python scripts/train_sf.py --cfg configs/humanoid_walk.yaml

    # Override any param from CLI
    python scripts/train_sf.py --cfg configs/humanoid_walk.yaml --learning_rate 1e-4 --num_workers 4

    # Without config file (all params on CLI)
    python scripts/train_sf.py --env beast --module_name HumanoidEnv \
        --project_path Projects/Humanoid/Humanoid.project --experiment my_run
"""

import argparse
import sys

import yaml
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from beastlab.sf import register_beast_envs, add_beast_args
from beastlab.models import register_beast_model


def _build_argv(yaml_path, cli_remaining):
    """Build argv list: yaml values first, then CLI overrides on top."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Convert yaml dict to flat --key value args
    yaml_argv = []
    for k, v in cfg.items():
        if isinstance(v, list):
            yaml_argv.append(f"--{k}")
            yaml_argv.extend(str(x) for x in v)
        elif isinstance(v, bool):
            yaml_argv.append(f"--{k}")
            yaml_argv.append(str(v))
        else:
            yaml_argv.append(f"--{k}")
            yaml_argv.append(str(v))

    # CLI args come after so they override yaml values
    return yaml_argv + cli_remaining


def main():
    register_beast_envs()
    register_beast_model()

    # Pre-parse --cfg before SF sees the args
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cfg", default=None)
    pre_args, remaining = pre.parse_known_args()

    if pre_args.cfg:
        argv = _build_argv(pre_args.cfg, remaining)
    else:
        argv = remaining

    parser, partial_cfg = parse_sf_args(argv=argv)
    add_beast_args(parser)
    cfg = parse_full_cfg(parser, argv=argv)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
