#!/usr/bin/env python3
"""Export a Sample Factory checkpoint to ONNX (Beast-compatible single file).

Produces a clean obs→actions model without SF's extra squeeze/unsqueeze ops.

Usage:
    python scripts/export_sf_onnx.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --output beast_ppo.onnx

    # Export best checkpoint
    python scripts/export_sf_onnx.py --env beast --module_name HumanoidEnv \
        --experiment humanoid_walk_v1 --output beast_ppo.onnx --load_checkpoint_kind best
"""

import sys

import torch
import torch.nn as nn
import numpy as np
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg, load_from_checkpoint
from sample_factory.enjoy import load_state_dict, make_env
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from beastlab.sf import register_beast_envs, add_beast_args


class BeastActorExporter(nn.Module):
    """Wraps SF actor_critic into a clean obs→actions module for Beast."""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs):
        obs_dict = {"obs": obs}
        normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs_dict)
        rnn_states = torch.zeros([obs.shape[0], get_rnn_size(self.actor_critic.cfg)], dtype=torch.float32)
        policy_outputs = self.actor_critic(normalized_obs, rnn_states)
        return policy_outputs["action_logits"]


def main():
    register_beast_envs()

    parser, _ = parse_sf_args(evaluation=True)
    add_beast_args(parser)
    parser.add_argument("--output", type=str, default="beast_ppo.onnx", help="Output ONNX path")
    cfg = parse_full_cfg(parser)
    cfg = load_from_checkpoint(cfg)

    env = make_env(cfg)
    env_info = extract_env_info(env, cfg)
    device = torch.device("cpu")

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)

    obs_size = env.observation_space["obs"].shape[0]
    action_size = env.action_space.shape[0]

    exporter = BeastActorExporter(actor_critic)
    exporter.eval()

    dummy_obs = torch.zeros(1, obs_size, dtype=torch.float32)

    torch.onnx.export(
        exporter,
        dummy_obs,
        cfg.output,
        export_params=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Ensure single file
    import onnx
    model = onnx.load(cfg.output)
    onnx.save(model, cfg.output, save_as_external_data=False)

    print(f"Exported to {cfg.output}")
    print(f"  obs shape:    ({obs_size},)")
    print(f"  action shape: ({action_size},)")

    # Verify
    try:
        import onnxruntime as ort
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(cfg.output)
        test_obs = np.zeros((1, obs_size), dtype=np.float32)
        onnx_out = sess.run(None, {"obs": test_obs})[0]
        print(f"  Output shape:  {onnx_out.shape}")

        with torch.no_grad():
            torch_out = exporter(dummy_obs).numpy()
        diff = np.abs(onnx_out - torch_out).max()
        print(f"  Verification: max diff = {diff:.2e} {'OK' if diff < 1e-5 else 'MISMATCH'}")
    except ImportError:
        print("  (install onnxruntime to verify)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
