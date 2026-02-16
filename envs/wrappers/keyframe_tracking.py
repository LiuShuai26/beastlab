"""Keyframe cycle tracking wrapper for gymnasium environments."""

import json
import math

import gymnasium as gym
import numpy as np


class KeyframeTrackingWrapper(gym.Wrapper):
    """Rewards the agent for cycling through a sequence of target joint-angle keyframes.

    Augments observation with current target joint angles and key body positions
    so the policy knows which pose to reach.

    Reward: r_track = w_j * exp(-k_j * mean_joint_diff) + w_b * exp(-k_b * mean_body_diff)

    Joint angles are read from ``info["joint_mapping"]`` every step.
    Key body positions are read from ``info["key_body_positions"]`` every step.
    """

    def __init__(
        self,
        env,
        keyframe_file,
        k_j=5.0,
        k_b=5.0,
        w_j=0.5,
        w_b=0.5,
        arrival_threshold=0.3,
        hold_steps=200,
        progress_bonus=10.0,
        smooth_coeff=0.01,
    ):
        super().__init__(env)

        with open(keyframe_file) as f:
            data = json.load(f)
        self.motion_name = data.get("motion_name", "unknown")
        self.keyframes = data["keyframes"]
        assert self.keyframes, "keyframe file must contain at least one keyframe"

        # Validate that env provides joint_mapping in info
        obs, info = env.reset()
        if "joint_mapping" not in info:
            raise ValueError("env.reset() info does not contain 'joint_mapping'")
        joint_mapping = info["joint_mapping"]

        # Separate joint angle keys from body position keys and metadata
        self.all_joint_names = sorted(
            {
                n
                for kf in self.keyframes
                for n in kf
                if not n.startswith("_")
                and not n.endswith("_x")
                and not n.endswith("_y")
            }
        )
        self.key_body_names = sorted(
            {
                n[:-2]
                for kf in self.keyframes
                for n in kf
                if n.endswith("_x") and not n.startswith("_")
            }
        )
        self.num_joints = len(self.all_joint_names)
        self.num_key_bodies = len(self.key_body_names)
        self.num_keyframes = len(self.keyframes)

        for kf in self.keyframes:
            for name in kf:
                if name.startswith("_") or name.endswith("_x") or name.endswith("_y"):
                    continue
                assert name in joint_mapping, (
                    f"Joint '{name}' in keyframe but not in joint_mapping. "
                    f"Available: {sorted(joint_mapping)}"
                )

        self.k_j = k_j
        self.k_b = k_b
        self.w_j = w_j
        self.w_b = w_b
        self.arrival_threshold = arrival_threshold
        self.hold_steps = hold_steps
        self.progress_bonus = progress_bonus
        self.smooth_coeff = smooth_coeff

        self.current_keyframe_idx = 0
        self.total_switches = 0
        self._hold_counter = 0
        self._prev_vel = None
        self._prev_angles = None
        self._ep_tracking_errors = []
        self._ep_r_track = 0.0
        self._ep_r_progress = 0.0
        self._ep_r_smooth = 0.0

        # Expand observation space: target angles + key body positions (x,y per body)
        extra_dim = self.num_joints + self.num_key_bodies * 2
        low = np.concatenate([env.observation_space.low, np.full(extra_dim, -np.inf)])
        high = np.concatenate([env.observation_space.high, np.full(extra_dim, np.inf)])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _joint_angles(self, joint_mapping):
        return np.array(
            [joint_mapping[n] for n in self.all_joint_names], dtype=np.float32
        )

    def _augment_obs(self, obs):
        kf = self.keyframes[self.current_keyframe_idx]
        target_angles = np.array(
            [kf.get(n, 0.0) for n in self.all_joint_names], dtype=np.float32
        )
        target_positions = np.array(
            [kf.get(f"{n}_{c}", 0.0) for n in self.key_body_names for c in ("x", "y")],
            dtype=np.float32,
        )
        return np.concatenate([obs, target_angles, target_positions])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_keyframe_idx = 0
        self.total_switches = 0
        self._hold_counter = 0
        self._prev_vel = None
        self._prev_angles = None
        self._ep_tracking_errors = []
        self._ep_r_track = 0.0
        self._ep_r_progress = 0.0
        self._ep_r_smooth = 0.0
        info["keyframe_idx"] = 0
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        joint_mapping = info["joint_mapping"]
        key_body_positions = info.get("key_body_positions", {})

        kf = self.keyframes[self.current_keyframe_idx]

        # Joint angle tracking: r_joint = exp(-k_j * mean(joint_diffs))
        joint_diffs = [
            abs(joint_mapping[n] - t)
            for n, t in kf.items()
            if not n.startswith("_") and not n.endswith("_x") and not n.endswith("_y")
        ]
        mean_joint_diff = float(np.mean(joint_diffs)) if joint_diffs else 0.0
        r_joint = math.exp(-self.k_j * mean_joint_diff)

        # Body position tracking: r_body = exp(-k_b * mean(body_diffs))
        body_diffs = []
        for name in self.key_body_names:
            tx = kf.get(f"{name}_x", 0.0)
            ty = kf.get(f"{name}_y", 0.0)
            cx = key_body_positions.get(f"{name}_x", 0.0)
            cy = key_body_positions.get(f"{name}_y", 0.0)
            body_diffs.append(math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))
        mean_body_diff = float(np.mean(body_diffs)) if body_diffs else 0.0
        r_body = math.exp(-self.k_b * mean_body_diff)

        # Combined tracking reward
        r_track = self.w_j * r_joint + self.w_b * r_body

        # Progress: must hold within threshold for hold_steps consecutive steps
        r_progress = 0.0
        # mean_diff = mean_joint_diff  # use joint diff for arrival detection
        # if mean_diff < self.arrival_threshold:
        #     self._hold_counter += 1
        #     if self._hold_counter >= self.hold_steps:
        #         r_progress = self.progress_bonus
        #         self.current_keyframe_idx = (self.current_keyframe_idx + 1) % self.num_keyframes
        #         self.total_switches += 1
        #         self._hold_counter = 0
        # else:
        #     self._hold_counter = 0

        # Smoothness penalty (dense, angular acceleration)
        angles = self._joint_angles(joint_mapping)
        r_smooth = 0.0
        if self._prev_angles is not None:
            vel = angles - self._prev_angles
            if self._prev_vel is not None:
                r_smooth = -self.smooth_coeff * float(
                    np.mean(np.abs(vel - self._prev_vel))
                )
            self._prev_vel = vel
        self._prev_angles = angles

        reward = r_track + r_progress + r_smooth

        # Accumulate episode-level stats
        self._ep_tracking_errors.append(mean_joint_diff)
        self._ep_r_track += r_track
        self._ep_r_progress += r_progress
        self._ep_r_smooth += r_smooth

        info["keyframe_idx"] = self.current_keyframe_idx
        info["tracking_error"] = mean_joint_diff
        info["body_tracking_error"] = mean_body_diff
        info["keyframe_switches"] = self.total_switches
        info["ep_mean_tracking_error"] = float(np.mean(self._ep_tracking_errors))
        info["ep_r_track"] = self._ep_r_track
        info["ep_r_progress"] = self._ep_r_progress
        info["ep_r_smooth"] = self._ep_r_smooth

        return self._augment_obs(obs), reward, terminated, truncated, info
