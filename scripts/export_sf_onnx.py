#!/usr/bin/env python3
"""Export a Sample Factory checkpoint to ONNX (Beast-compatible single file).

Produces a clean obs→actions model matching Beast's expected format.

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
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from beastlab.sf import register_beast_envs, add_beast_args


def build_plain_actor(actor_critic, obs_size, action_size):
    """Extract weights from SF actor_critic and build a plain nn.Sequential."""
    sd = actor_critic.state_dict()

    # Collect encoder MLP layers (encoder.encoders.obs.mlp_head.N.{weight,bias})
    # SF Sequential uses indices 0, 2, 4... for Linear and 1, 3, 5... for activations
    encoder_layers = []
    prev_linear = False
    for i in range(20):  # scan up to 20 sub-modules
        w_key = f"encoder.encoders.obs.mlp_head.{i}.weight"
        b_key = f"encoder.encoders.obs.mlp_head.{i}.bias"
        if w_key in sd:
            if prev_linear:
                encoder_layers.append(nn.ELU())
            linear = nn.Linear(sd[w_key].shape[1], sd[w_key].shape[0])
            linear.weight.data = sd[w_key]
            linear.bias.data = sd[b_key]
            encoder_layers.append(linear)
            prev_linear = True
    # SF applies activation after every linear layer, including the last
    if encoder_layers:
        encoder_layers.append(nn.ELU())

    # Decoder layers (decoder.mlp.N.{weight,bias})
    decoder_layers = []
    prev_linear = False
    for i in range(20):
        w_key = f"decoder.mlp.{i}.weight"
        b_key = f"decoder.mlp.{i}.bias"
        if w_key in sd:
            if prev_linear:
                decoder_layers.append(nn.ELU())
            linear = nn.Linear(sd[w_key].shape[1], sd[w_key].shape[0])
            linear.weight.data = sd[w_key]
            linear.bias.data = sd[b_key]
            decoder_layers.append(linear)
            prev_linear = True

    # Action head (action_parameterization.distribution_linear.weight/bias)
    action_w = sd["action_parameterization.distribution_linear.weight"]
    action_b = sd["action_parameterization.distribution_linear.bias"]
    # This outputs [mean, log_std] so shape is (action_size*2, hidden)
    # We only want the mean (first action_size rows)
    action_head = nn.Linear(action_w.shape[1], action_size)
    action_head.weight.data = action_w[:action_size]
    action_head.bias.data = action_b[:action_size]

    # Obs normalization (try nested key path first, then flat)
    obs_mean = sd.get(
        "obs_normalizer.running_mean_std.running_mean_std.obs.running_mean",
        sd.get("obs_normalizer.running_mean_std.running_mean", torch.zeros(obs_size)),
    )
    obs_var = sd.get(
        "obs_normalizer.running_mean_std.running_mean_std.obs.running_var",
        sd.get("obs_normalizer.running_mean_std.running_var", torch.ones(obs_size)),
    )

    class PlainActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_mean = nn.Parameter(obs_mean.float(), requires_grad=False)
            self.obs_var = nn.Parameter(obs_var.float(), requires_grad=False)
            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
            self.action_head = action_head

        def forward(self, obs):
            # Normalize obs
            x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
            x = torch.clamp(x, -5.0, 5.0)
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.action_head(x)
            return x

    return PlainActor()


def main():
    register_beast_envs()

    parser, _ = parse_sf_args(evaluation=True)
    add_beast_args(parser)
    parser.add_argument("--output", type=str, default="beast_ppo.onnx", help="Output ONNX path")
    cfg = parse_full_cfg(parser)
    cfg = load_from_checkpoint(cfg)

    env = make_env(cfg)
    device = torch.device("cpu")

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)

    obs_size = env.observation_space["obs"].shape[0]
    action_size = env.action_space.shape[0]

    actor = build_plain_actor(actor_critic, obs_size, action_size)
    actor.eval()

    dummy_obs = torch.zeros(1, obs_size, dtype=torch.float32)

    # Verify outputs match before export
    with torch.no_grad():
        plain_out = actor(dummy_obs)
        print(f"Plain actor output shape: {plain_out.shape}")

    torch.onnx.export(
        actor,
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
            torch_out = actor(dummy_obs).numpy()
        diff = np.abs(onnx_out - torch_out).max()
        print(f"  Verification: max diff = {diff:.2e} {'OK' if diff < 1e-4 else 'MISMATCH'}")
    except ImportError:
        print("  (install onnxruntime to verify)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
