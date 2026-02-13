#!/usr/bin/env python3
"""Export a Stable Baselines3 model to ONNX.

Prerequisites:
    pip install stable-baselines3 onnx

Usage:
    python export_onnx.py --model beast_ppo.zip --output beast_ppo.onnx
"""

import argparse

import torch
import numpy as np
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO model to ONNX")
    parser.add_argument("--model", default="beast_ppo.zip", help="Path to SB3 .zip model")
    parser.add_argument("--output", default="beast_ppo.onnx", help="Output ONNX path")
    args = parser.parse_args()

    model = PPO.load(args.model)
    policy = model.policy
    policy.eval()

    obs_size = model.observation_space.shape[0]
    dummy_obs = torch.zeros(1, obs_size, dtype=torch.float32)

    # SB3 MlpPolicy stores the actor network in policy.mlp_extractor + policy.action_net
    # Export just the deterministic action (mean) path
    class ActorWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            features = self.mlp_extractor.forward_actor(obs)
            return self.action_net(features)

    actor = ActorWrapper(policy)
    actor.eval()

    torch.onnx.export(
        actor,
        dummy_obs,
        args.output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    print(f"Exported to {args.output}")
    print(f"  obs shape:    ({obs_size},)")
    print(f"  action shape: ({model.action_space.shape[0]},)")

    # Quick verification
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(args.output)
        test_obs = np.zeros((1, obs_size), dtype=np.float32)
        onnx_out = sess.run(None, {"obs": test_obs})[0]

        with torch.no_grad():
            torch_out = actor(dummy_obs).numpy()

        diff = np.abs(onnx_out - torch_out).max()
        print(f"  Verification: max diff = {diff:.2e} {'OK' if diff < 1e-5 else 'MISMATCH'}")
    except ImportError:
        print("  (install onnxruntime to verify: pip install onnxruntime)")


if __name__ == "__main__":
    main()
