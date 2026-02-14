#!/usr/bin/env python3
"""Evaluate a Sample Factory checkpoint using SF's built-in enjoy.

Usage:
    python scripts/enjoy_sf.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --no_render --max_num_frames 5000

    # Best checkpoint
    python scripts/enjoy_sf.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --no_render --load_checkpoint_kind best
"""

import sys
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.enjoy import enjoy

from beastlab.sf import register_beast_envs, add_beast_args
from beastlab.models import register_beast_model


def main():
    register_beast_envs()
    register_beast_model()

    parser, _ = parse_sf_args(evaluation=True)
    add_beast_args(parser)
    cfg = parse_full_cfg(parser)
    status, _ = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
