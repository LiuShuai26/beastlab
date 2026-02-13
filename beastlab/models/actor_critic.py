"""
Custom Sample Factory encoder for BeastLab humanoid environments.

Currently uses SF's built-in MLP encoder. This file exists as the extension
point for custom architectures (e.g., adding discriminator heads for AMP).
"""


def register_beast_model():
    """Register custom encoder with Sample Factory. Currently a no-op placeholder."""
    # Will be implemented when we need custom network architecture.
    # For now, SF's built-in mlp_mujoco encoder works fine.
    pass
