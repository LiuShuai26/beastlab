#!/usr/bin/env python3
"""Export a Sample Factory checkpoint to ONNX.

Usage:
    python scripts/export_sf_onnx.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --output beast_ppo.onnx

    # Export best checkpoint instead of latest
    python scripts/export_sf_onnx.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --output beast_ppo.onnx --load_checkpoint_kind best
"""

import sys

from sample_factory.export_onnx import export_onnx
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg

from beastlab.sf import register_beast_envs, add_beast_args


def main():
    register_beast_envs()

    parser, _ = parse_sf_args(evaluation=True)
    add_beast_args(parser)
    parser.add_argument("--output", type=str, default="beast_ppo.onnx", help="Output ONNX path")
    cfg = parse_full_cfg(parser)

    status = export_onnx(cfg, cfg.output)

    # Ensure single file (no external data)
    import onnx
    model = onnx.load(cfg.output)
    onnx.save(model, cfg.output, save_as_external_data=False)

    print(f"Exported to {cfg.output}")
    return status


if __name__ == "__main__":
    sys.exit(main())
