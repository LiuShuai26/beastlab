"""
Beast ML Training Client

Connects to the Beast editor's MLServer via TCP binary protocol and provides
a gymnasium-compatible environment for reinforcement learning training.

Usage:
    from beastlab import BeastClientEnv, BeastDisconnectedError

    # Using context manager (recommended - handles cleanup on Ctrl+C)
    try:
        with BeastClientEnv(port=5555) as env:
            obs, info = env.reset()
            for _ in range(1000):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
    except KeyboardInterrupt:
        print("Training interrupted")
    except BeastDisconnectedError:
        print("Lost connection to Beast")

    # Or manual cleanup
    env = BeastClientEnv()
    try:
        # ... training loop ...
    finally:
        env.close()
"""

import atexit
import socket
import struct
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        _HAS_GYM = True
    except ImportError:
        _HAS_GYM = False


# ---------------------------------------------------------------------------
# Binary protocol (must match MLProtocol.h)
# ---------------------------------------------------------------------------

class _MsgType:
    # Client -> Server
    HELLO = 0x01
    RESET = 0x02
    STEP  = 0x03
    CLOSE = 0x04
    # Server -> Client
    CONFIG      = 0x10
    STEP_RESULT = 0x11
    ERROR       = 0xFF


_HEADER_FMT = '<II'  # type (uint32), size (uint32)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BeastDisconnectedError(ConnectionError):
    """Raised when connection to Beast MLServer is lost."""
    pass


# ---------------------------------------------------------------------------
# BeastClientEnv — single-env gymnasium wrapper
# ---------------------------------------------------------------------------

_BaseClass = gym.Env if _HAS_GYM else object


class BeastClientEnv(_BaseClass):
    """Gymnasium-compatible environment that connects to Beast editor via TCP.

    Connects to the Beast editor's built-in MLServer (Habitat) and exposes
    the simulation as a standard gymnasium environment for RL training.

    Args:
        host: MLServer hostname (default: 127.0.0.1).
        port: MLServer port (default: 5555).
        timeout: Socket timeout in seconds (default: 30).
        auto_connect: Whether to connect automatically on init (default: True).
        joint_mapping: Optional dict mapping joint names to obs indices (sin/cos pairs).
        key_body_mapping: Optional dict mapping key body names to obs indices (x/y pairs).
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: float = 30.0,
        auto_connect: bool = True,
        joint_mapping: Optional[Dict[str, int]] = None,
        key_body_mapping: Optional[Dict[str, int]] = None,
    ):
        if _HAS_GYM and isinstance(self, gym.Env):
            super().__init__()

        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self.connected = False

        # Config (populated after hello)
        self.observation_size: int = 0
        self.continuous_action_size: int = 0
        self.discrete_branches: List[int] = []
        self.num_reward_groups: int = 1

        # Project-specific mappings (injected into info dicts)
        self.joint_mapping: Dict[str, int] = joint_mapping or {}
        self.key_body_mapping: Dict[str, int] = key_body_mapping or {}

        # Spaces (set after connect)
        self.observation_space = None
        self.action_space = None

        atexit.register(self._cleanup)

        if auto_connect:
            self.connect()

    def _cleanup(self):
        """Cleanup handler for atexit."""
        if self.connected:
            self.close()

    def connect(self) -> bool:
        """Connect to the Beast MLServer and perform handshake."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.settimeout(self._timeout)
            self._sock.connect((self._host, self._port))
            self.connected = True

            # Handshake: HELLO -> CONFIG
            self._send(_MsgType.HELLO)
            msg_type, payload = self._recv()
            if msg_type != _MsgType.CONFIG:
                raise ConnectionError(f"Expected Config ({_MsgType.CONFIG:#x}), got {msg_type:#x}")
            self._parse_config(payload)

            # Set up gymnasium spaces
            self._setup_spaces()

            print(f"Connected to Beast MLServer at {self._host}:{self._port}")
            print(f"  Observation size: {self.observation_size}")
            print(f"  Continuous actions: {self.continuous_action_size}")
            if self.discrete_branches:
                print(f"  Discrete branches: {self.discrete_branches}")
            if self.num_reward_groups > 1:
                print(f"  Reward groups: {self.num_reward_groups}")

            return True

        except Exception as e:
            print(f"Failed to connect to Beast MLServer: {e}")
            self._mark_disconnected()
            return False

    def _parse_config(self, payload: bytes):
        """Parse Config payload from server."""
        offset = 0
        self.observation_size = struct.unpack_from('<i', payload, offset)[0]; offset += 4
        self.continuous_action_size = struct.unpack_from('<i', payload, offset)[0]; offset += 4
        num_branches = struct.unpack_from('<i', payload, offset)[0]; offset += 4
        self.discrete_branches = []
        for _ in range(num_branches):
            bs = struct.unpack_from('<i', payload, offset)[0]; offset += 4
            self.discrete_branches.append(bs)
        # numRewardGroups appended after branchSizes (backward-compatible: default 1)
        if offset < len(payload):
            self.num_reward_groups = struct.unpack_from('<i', payload, offset)[0]; offset += 4
        else:
            self.num_reward_groups = 1

    def _setup_spaces(self):
        """Set up gymnasium observation and action spaces from config."""
        if not _HAS_GYM:
            return

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_size,), dtype=np.float32,
        )

        # Action space
        has_discrete = len(self.discrete_branches) > 0
        has_continuous = self.continuous_action_size > 0

        if has_discrete and has_continuous:
            # Hybrid action space
            discrete_space = (spaces.Discrete(self.discrete_branches[0])
                              if len(self.discrete_branches) == 1
                              else spaces.MultiDiscrete(self.discrete_branches))
            continuous_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.continuous_action_size,),
                dtype=np.float32,
            )
            self.action_space = spaces.Dict({
                "discrete": discrete_space,
                "continuous": continuous_space,
            })
        elif has_continuous:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.continuous_action_size,), dtype=np.float32,
            )
        elif has_discrete:
            if len(self.discrete_branches) == 1:
                self.action_space = spaces.Discrete(self.discrete_branches[0])
            else:
                self.action_space = spaces.MultiDiscrete(self.discrete_branches)
        else:
            self.action_space = spaces.Discrete(2)

    def _mark_disconnected(self):
        """Mark the environment as disconnected."""
        self.connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Returns:
            observation: Initial observation.
            info: Additional information dict.
        """
        if _HAS_GYM and seed is not None and isinstance(self, gym.Env):
            super().reset(seed=seed)

        if not self.connected:
            self.connect()

        self._send(_MsgType.RESET)
        msg_type, payload = self._recv()
        if msg_type != _MsgType.STEP_RESULT:
            raise BeastDisconnectedError(f"Expected StepResult, got {msg_type:#x}")

        obs, _, _, _, info = self._parse_step_result(payload)
        return obs, info

    def step(
        self,
        actions: Union[int, np.ndarray, List, Dict],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with the given actions.

        Args:
            actions: Continuous actions as numpy array, or dict for hybrid.

        Returns:
            obs: Observation.
            reward: Reward.
            terminated: True if episode terminated.
            truncated: True if episode truncated.
            info: Additional info dict.
        """
        if not self.connected:
            raise BeastDisconnectedError("Not connected to server")

        # Build action bytes (continuous float array for the binary protocol)
        action_bytes = self._format_actions(actions)
        self._send(_MsgType.STEP, action_bytes)

        msg_type, payload = self._recv()
        if msg_type != _MsgType.STEP_RESULT:
            raise BeastDisconnectedError(f"Expected StepResult, got {msg_type:#x}")

        obs, rewards, terminated_arr, truncated_arr, info = self._parse_step_result(payload)

        reward = float(rewards[0])
        terminated = bool(terminated_arr[0])
        truncated = bool(truncated_arr[0])

        return obs, reward, terminated, truncated, info

    def _format_actions(self, actions) -> bytes:
        """Convert actions to binary payload (float32 array)."""
        if isinstance(actions, dict):
            # Hybrid action space: extract continuous part
            continuous = actions.get("continuous", np.zeros(self.continuous_action_size, dtype=np.float32))
            actions = np.asarray(continuous, dtype=np.float32)
        else:
            actions = np.asarray(actions, dtype=np.float32)
        return actions.ravel().tobytes()

    def close(self):
        """Gracefully close the connection."""
        if self._sock:
            try:
                self._send(_MsgType.CLOSE)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self.connected:
            self.connected = False
            print("Disconnected from Beast MLServer")

    def render(self):
        """Render is handled by Beast editor."""
        pass

    # -------------------------------------------------------------------
    # Binary protocol helpers
    # -------------------------------------------------------------------

    def _parse_step_result(self, payload: bytes):
        """Parse StepResult payload into numpy arrays."""
        obs_size = self.observation_size
        nrg = self.num_reward_groups
        offset = 0

        obs = np.frombuffer(payload, dtype=np.float32, count=obs_size, offset=offset).copy()
        offset += obs_size * 4

        # Read N reward group floats
        reward_groups = np.frombuffer(payload, dtype=np.float32, count=nrg, offset=offset).copy()
        offset += nrg * 4

        # Scalar reward = sum of all groups (gym-compatible)
        rewards = np.array([reward_groups.sum()], dtype=np.float32)

        terminated = np.array([payload[offset] != 0], dtype=bool)
        offset += 1
        truncated = np.array([payload[offset] != 0], dtype=bool)

        info = {}
        if nrg > 1:
            info["reward_groups"] = reward_groups
        if self.joint_mapping:
            info["joint_mapping"] = {
                name: float(np.arctan2(obs[idx], obs[idx + 1]))
                for name, idx in self.joint_mapping.items()
            }
        if self.key_body_mapping:
            kbp = {}
            for name, idx in self.key_body_mapping.items():
                kbp[f"{name}_x"] = float(obs[idx])
                kbp[f"{name}_y"] = float(obs[idx + 1])
            info["key_body_positions"] = kbp
        return obs, rewards, terminated, truncated, info

    def _send(self, msg_type: int, payload: bytes = b''):
        """Send a message with header."""
        if not self._sock:
            raise BeastDisconnectedError("Not connected")
        try:
            header = struct.pack(_HEADER_FMT, msg_type, len(payload))
            self._sock.sendall(header + payload)
        except (socket.error, OSError) as e:
            self._mark_disconnected()
            raise BeastDisconnectedError(f"Connection lost: {e}")

    def _recv(self) -> Tuple[int, bytes]:
        """Receive a complete message. Returns (msg_type, payload_bytes)."""
        try:
            header_data = self._recv_exact(_HEADER_SIZE)
            msg_type, payload_size = struct.unpack(_HEADER_FMT, header_data)
            payload = self._recv_exact(payload_size) if payload_size > 0 else b''
            return msg_type, payload
        except (socket.error, OSError) as e:
            self._mark_disconnected()
            raise BeastDisconnectedError(f"Connection lost: {e}")

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                self._mark_disconnected()
                raise BeastDisconnectedError("Connection closed by server")
            buf.extend(chunk)
        return bytes(buf)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_beast_env(
    host: str = "127.0.0.1",
    port: int = 5555,
    joint_mapping: Optional[Dict[str, int]] = None,
    key_body_mapping: Optional[Dict[str, int]] = None,
) -> BeastClientEnv:
    """Create a Beast environment.

    Args:
        host: MLServer host.
        port: MLServer port.
        joint_mapping: Optional dict mapping joint names to obs indices (sin/cos pairs).
        key_body_mapping: Optional dict mapping key body names to obs indices (x/y pairs).

    Returns:
        BeastClientEnv instance.
    """
    return BeastClientEnv(host=host, port=port, joint_mapping=joint_mapping,
                          key_body_mapping=key_body_mapping)


# ---------------------------------------------------------------------------
# Test script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Beast ML Client - Environment Test")
    print("=" * 50)
    print("Make sure Beast editor is running with Training enabled in the Habitat panel!")
    print("Press Ctrl+C to stop.\n")

    try:
        with BeastClientEnv() as env:
            if not env.connected:
                print("Failed to connect. Is Beast running with Training enabled?")
                exit(1)

            print(f"\nSpaces: obs={env.observation_space}, act={env.action_space}")

            # Test reset
            print("\nTesting reset...")
            obs, info = env.reset()
            print(f"Initial observation shape: {obs.shape}")

            # Test steps
            print("\nTesting steps...")
            total_reward = 0.0
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"  Step {i+1}: reward={reward:.4f}, term={terminated}, trunc={truncated}")

                if terminated or truncated:
                    print("  Episode ended! Resetting...")
                    obs, info = env.reset()
                    total_reward = 0.0

            print(f"\nTotal reward: {total_reward:.4f}")
            print("Test complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except BeastDisconnectedError as e:
        print(f"\nDisconnected from server: {e}")
    except Exception as e:
        print(f"\nError: {e}")
