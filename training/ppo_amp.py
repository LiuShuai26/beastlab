# PPO with Adversarial Motion Priors (AMP) for stylized locomotion.
#
# Discriminator judges (s_t, s_{t+1}) transition pairs.  AMP state features
# are extracted directly from the raw observation by index — no obs_config
# file needed, no info-dict parsing.
#
# Required:
#   --keyframe-file : path to keyframe JSON (e.g. fight_walk_3.json)
#
# Observation layout (from Brain.cpp, 76 elements):
#   [0]     pelvis_y
#   [1-2]   sin/cos(pelvis_angle)
#   [3-5]   pelvis vx, vy, angular_vel
#   [6-29]  joint sin/cos (12 joints × 2)
#   [30-41] joint angular velocities (12)
#   [42-51] key body positions in pelvis local frame (5 × 2)
#   [52-63] previous actions (12)
#   [64-75] energy levels (12)
#
# AMP state = obs[0:1] + obs[6:30] + obs[42:52]  →  35 dims
# Transition = concat(state_t, state_{t+1})       →  70 dims

import atexit
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from beastlab.amp import (
    AMPDiscriminator,
    AMPMotionBuffer,
    compute_disc_loss,
    compute_style_reward,
)
from beastlab.client import BeastClientEnv
from beastlab.env_loader import make_beast_gym


# ---------------------------------------------------------------------------
# Humanoid observation → AMP state index map  (must match Brain.cpp)
# ---------------------------------------------------------------------------

# Joint order — must match Brain.cpp JOINT_TAGS
JOINT_ORDER = [
    "abdomen",
    "neck",
    "right_shoulder",
    "right_elbow",
    "left_shoulder",
    "left_elbow",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
]

# Key body order — must match HumanoidConfig.h KEY_BODY_TAGS
BODY_ORDER = [
    "head",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
]

# Raw observation indices for AMP state extraction
AMP_PELVIS_Y = slice(0, 1)  # 1 dim
AMP_JOINT_SINCOS = slice(6, 30)  # 24 dims
AMP_BODY_POS = slice(42, 52)  # 10 dims
AMP_OBS_DIM = 1 + 24 + 10  # 35


def extract_amp_from_raw_obs(raw_obs):
    """Extract 35-dim AMP state directly from the raw observation vector.

    Args:
        raw_obs: numpy array of shape (obs_size,) — unnormalised.

    Returns:
        numpy array of shape (35,).
    """
    return np.concatenate(
        [
            raw_obs[AMP_PELVIS_Y],
            raw_obs[AMP_JOINT_SINCOS],
            raw_obs[AMP_BODY_POS],
        ]
    )


# ---------------------------------------------------------------------------
# Wrapper: capture raw obs before NormalizeObservation
# ---------------------------------------------------------------------------


class RawObsWrapper(gym.Wrapper):
    """Stores the unnormalised observation in info['raw_obs'].

    Must sit *before* NormalizeObservation in the wrapper stack so the
    raw values are captured before scaling.
    """

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info["raw_obs"] = obs.copy()
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["raw_obs"] = obs.copy()
        return obs, info


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HumanoidEnv"
    """gym env id, or Beast .so module name"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 512
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    lcp_coef: float = 0.0
    """coefficient for Lipschitz constraint penalty (gradient penalty on actor)"""
    hidden_dim: int = 256
    """hidden layer size for actor and critic networks"""

    # AMP specific
    keyframe_file: str = ""
    """path to keyframe JSON file (required)"""
    disc_lr: float = 5e-4
    """discriminator learning rate"""
    disc_hidden_dim: int = 256
    """discriminator hidden layer size"""
    disc_num_layers: int = 2
    """number of discriminator hidden layers"""
    disc_grad_penalty_coef: float = 5.0
    """gradient penalty coefficient for discriminator"""
    disc_weight_decay: float = 1e-4
    """discriminator optimizer weight decay"""
    disc_update_epochs: int = 1
    """number of discriminator update epochs per PPO iteration"""
    task_reward_weight: float = 0.1
    """weight for task reward group in advantage combination"""
    style_reward_weight: float = 0.9
    """weight for style reward group in advantage combination"""

    # Editor training
    editor: bool = False
    """connect to Beast editor via TCP instead of headless .so"""
    editor_port: int = 5555
    """Beast editor MLServer port"""

    # Computed at runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def make_env(
    env_id, idx, capture_video, run_name, gamma, editor=False, editor_port=5555
):
    def thunk():
        if editor:
            env = BeastClientEnv(port=editor_port)
        else:
            try:
                env = make_beast_gym(env_id)
            except FileNotFoundError:
                if capture_video and idx == 0:
                    env = gym.make(env_id, render_mode="rgb_array")
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
                else:
                    env = gym.make(env_id)
                env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = RawObsWrapper(env)  # <-- capture raw obs
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NormalizedActor(nn.Module):
    """Actor with baked-in observation normalisation (for ONNX export)."""

    def __init__(self, actor_mean, mean, var, epsilon=1e-8):
        super().__init__()
        self.actor_mean = actor_mean
        self.register_buffer("obs_mean", mean)
        self.register_buffer("obs_var", var)
        self.epsilon = epsilon

    def forward(self, obs):
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -10.0, 10.0)
        return self.actor_mean(obs)


class Agent(nn.Module):
    """PPO agent with two critics: task reward + style reward."""

    def __init__(self, envs, hidden_dim=256):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        G = 2  # task + style

        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    layer_init(nn.Linear(obs_dim, hidden_dim)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden_dim, hidden_dim)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden_dim, 1), std=1.0),
                )
                for _ in range(G)
            ]
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_values(self, x):
        """Returns (B, 2) tensor of per-group values."""
        return torch.cat([c(x) for c in self.critics], dim=-1)

    def get_action_and_values(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.get_values(x),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.keyframe_file, "--keyframe-file is required for AMP training"

    G = 2  # reward groups: 0 = task, 1 = style
    reward_weights = [args.task_reward_weight, args.style_reward_weight]

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    reward_weights_t = torch.tensor(reward_weights, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # AMP: motion buffer (provides transition pairs)
    # ------------------------------------------------------------------
    motion_buffer = AMPMotionBuffer(
        args.keyframe_file,
        joint_order=JOINT_ORDER,
        body_order=BODY_ORDER,
        device=device,
    )
    amp_obs_dim = motion_buffer.obs_dim
    amp_transition_dim = motion_buffer.transition_dim
    assert amp_obs_dim == AMP_OBS_DIM, (
        f"Motion buffer obs_dim ({amp_obs_dim}) != expected ({AMP_OBS_DIM}). "
        f"Check JOINT_ORDER / BODY_ORDER."
    )

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                editor=args.editor,
                editor_port=args.editor_port,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # ------------------------------------------------------------------
    # Networks + optimisers
    # ------------------------------------------------------------------
    agent = Agent(envs, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Discriminator takes transition pairs (s_t, s_{t+1}) as input
    discriminator = AMPDiscriminator(
        amp_transition_dim,
        hidden_dim=args.disc_hidden_dim,
        num_layers=args.disc_num_layers,
    ).to(device)
    disc_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.disc_lr,
        weight_decay=args.disc_weight_decay,
    )

    # ------------------------------------------------------------------
    # Checkpoint saving (on exit / Ctrl-C / SIGTERM)
    # ------------------------------------------------------------------
    _save_state = {"done": False}

    def save_checkpoint():
        if _save_state["done"] or not args.save_model:
            return
        _save_state["done"] = True
        run_dir = f"runs/{run_name}"
        os.makedirs(run_dir, exist_ok=True)

        model_path = f"{run_dir}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"\nmodel saved to {model_path}")

        disc_path = f"{run_dir}/{args.exp_name}.disc_model"
        torch.save(discriminator.state_dict(), disc_path)
        print(f"discriminator saved to {disc_path}")

        # ONNX export (actor only, with normalisation baked in)
        sub_env = envs.envs[0]
        obs_mean, obs_var = None, None
        while sub_env is not None:
            if isinstance(sub_env, gym.wrappers.NormalizeObservation):
                obs_mean = torch.tensor(
                    sub_env.obs_rms.mean, dtype=torch.float32, device=device
                )
                obs_var = torch.tensor(
                    sub_env.obs_rms.var, dtype=torch.float32, device=device
                )
                break
            sub_env = getattr(sub_env, "env", None)

        obs_dim = agent.actor_mean[0].in_features
        dummy_input = torch.zeros(1, obs_dim, device=device)
        if obs_mean is not None:
            export_model = NormalizedActor(agent.actor_mean, obs_mean, obs_var).to(
                device
            )
        else:
            export_model = agent.actor_mean
        export_model.eval()
        onnx_path = f"{run_dir}/{args.exp_name}.onnx"
        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_path,
            input_names=["obs"],
            output_names=["action_mean"],
            dynamic_axes={"obs": {0: "batch"}, "action_mean": {0: "batch"}},
        )
        print(f"onnx saved to {onnx_path}")

    atexit.register(save_checkpoint)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_g = torch.zeros((args.num_steps, args.num_envs, G)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_g = torch.zeros((args.num_steps, args.num_envs, G)).to(device)

    # AMP observation buffer — per-frame states for building transitions
    amp_obs_buf = torch.zeros((args.num_steps, args.num_envs, amp_obs_dim)).to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ==============================================================
        # Rollout phase
        # ==============================================================
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, vals_g = agent.get_action_and_values(next_obs)
                values_g[step] = vals_g
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)

            # ---- Task reward (group 0): env reward ----
            for ei in range(args.num_envs):
                r = reward[ei] if hasattr(reward, "__getitem__") else reward
                rewards_g[step, ei, 0] = float(r)

            # ---- AMP state extraction from raw obs ----
            # RawObsWrapper stores unnormalised obs in info["raw_obs"].
            # SyncVectorEnv may return an object-dtype array on auto-reset
            # boundaries, so we cast each element to float32 individually.
            raw_obs_all = infos.get("raw_obs")
            if raw_obs_all is not None:
                for ei in range(args.num_envs):
                    raw = raw_obs_all[ei] if raw_obs_all.ndim >= 1 else raw_obs_all
                    raw = np.asarray(raw, dtype=np.float32)
                    amp_feat = extract_amp_from_raw_obs(raw)
                    amp_obs_buf[step, ei] = torch.from_numpy(amp_feat).to(device)

            next_obs_np = np.array(next_obs_np)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # ==============================================================
        # Build transition pairs from AMP observation buffer
        # ==============================================================
        # Transition t = concat(s_t, s_{t+1}).  Valid when no episode
        # boundary between t and t+1 — dones[t+1] == 0.
        trans_s = amp_obs_buf[:-1]  # (T-1, E, D)
        trans_s_next = amp_obs_buf[1:]  # (T-1, E, D)
        policy_transitions = torch.cat([trans_s, trans_s_next], dim=-1)  # (T-1, E, 2D)

        valid_mask = dones[1:] == 0  # (T-1, E)
        flat_transitions = policy_transitions.reshape(-1, amp_transition_dim)
        flat_valid = valid_mask.reshape(-1).bool()
        valid_transitions = flat_transitions[flat_valid]

        # ==============================================================
        # Compute style rewards (group 1) using current discriminator
        # ==============================================================
        with torch.no_grad():
            disc_logits_all = discriminator(
                motion_buffer.normalize(
                    policy_transitions.reshape(-1, amp_transition_dim)
                )
            )
            style_r_flat = compute_style_reward(disc_logits_all)
            style_r_flat = style_r_flat * flat_valid.float()
            style_r = style_r_flat.reshape(args.num_steps - 1, args.num_envs)

            rewards_g[:, :, 1] = 0.0
            rewards_g[:-1, :, 1] = style_r

        # ==============================================================
        # Per-group GAE
        # ==============================================================
        with torch.no_grad():
            next_values_g = agent.get_values(next_obs)
            advantages_g = torch.zeros_like(rewards_g).to(device)

            for g in range(G):
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_values_g[:, g]
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values_g[t + 1, :, g]
                    delta = (
                        rewards_g[t, :, g]
                        + args.gamma * nextvalues * nextnonterminal
                        - values_g[t, :, g]
                    )
                    advantages_g[t, :, g] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )

            returns_g = advantages_g + values_g

            combined_advantages = torch.zeros(
                args.num_steps, args.num_envs, device=device
            )
            for g in range(G):
                ag = advantages_g[:, :, g]
                ag_norm = (ag - ag.mean()) / (ag.std() + 1e-8)
                combined_advantages += reward_weights_t[g] * ag_norm

        # ==============================================================
        # Flatten the batch (for PPO update)
        # ==============================================================
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = combined_advantages.reshape(-1)
        b_returns_g = returns_g.reshape(-1, G)
        b_values_g = values_g.reshape(-1, G)

        # ==============================================================
        # PPO policy + value update
        # ==============================================================
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]

                _, newlogprob, entropy, newvalues_g = agent.get_action_and_values(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.0
                for g in range(G):
                    nv = newvalues_g[:, g]
                    if args.clip_vloss:
                        v_loss_unclipped = (nv - b_returns_g[mb_inds, g]) ** 2
                        v_clipped = b_values_g[mb_inds, g] + torch.clamp(
                            nv - b_values_g[mb_inds, g],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss += (
                            0.5
                            * torch.max(
                                v_loss_unclipped,
                                (v_clipped - b_returns_g[mb_inds, g]) ** 2,
                            ).mean()
                        )
                    else:
                        v_loss += 0.5 * ((nv - b_returns_g[mb_inds, g]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if args.lcp_coef > 0:
                    mb_obs_grad = b_obs[mb_inds].detach().requires_grad_(True)
                    action_mean = agent.actor_mean(mb_obs_grad)
                    grad = torch.autograd.grad(
                        action_mean.sum(), mb_obs_grad, create_graph=True
                    )[0]
                    grad_penalty = grad.norm(2, dim=-1).mean()
                    loss = loss + args.lcp_coef * grad_penalty

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ==============================================================
        # Discriminator update (on valid transition pairs)
        # ==============================================================
        disc_losses = []
        disc_gp_losses = []
        disc_real_acc = []
        disc_fake_acc = []

        n_valid = valid_transitions.shape[0]
        if n_valid > 0:
            disc_inds = np.arange(n_valid)
            disc_mb_size = max(1, n_valid // args.num_minibatches)
            for _disc_epoch in range(args.disc_update_epochs):
                np.random.shuffle(disc_inds)
                for start in range(0, n_valid, disc_mb_size):
                    mb_disc_inds = disc_inds[start : start + disc_mb_size]
                    mb_size = len(mb_disc_inds)

                    raw_fake = valid_transitions[mb_disc_inds]
                    raw_real = motion_buffer.sample(mb_size)
                    fake_trans = motion_buffer.normalize(raw_fake)
                    real_trans = motion_buffer.normalize(raw_real)

                    disc_real_logits = discriminator(real_trans)
                    disc_fake_logits = discriminator(fake_trans.detach())

                    disc_loss = compute_disc_loss(disc_real_logits, disc_fake_logits)
                    gp_loss = discriminator.compute_grad_penalty(real_trans)
                    total_disc_loss = disc_loss + args.disc_grad_penalty_coef * gp_loss

                    disc_optimizer.zero_grad()
                    total_disc_loss.backward()
                    nn.utils.clip_grad_norm_(
                        discriminator.parameters(), args.max_grad_norm
                    )
                    disc_optimizer.step()

                    disc_losses.append(disc_loss.item())
                    disc_gp_losses.append(gp_loss.item())
                    with torch.no_grad():
                        disc_real_acc.append(
                            (disc_real_logits > 0).float().mean().item()
                        )
                        disc_fake_acc.append(
                            (disc_fake_logits < 0).float().mean().item()
                        )

                    # Debug: print once per iteration (first minibatch of first epoch)
                    if _disc_epoch == 0 and start == 0:
                        with torch.no_grad():
                            print(
                                f"  [disc debug] raw_real  range: [{raw_real.min():.3f}, {raw_real.max():.3f}]  mean: {raw_real.mean():.3f}  std: {raw_real.std():.3f}"
                            )
                            print(
                                f"  [disc debug] raw_fake  range: [{raw_fake.min():.3f}, {raw_fake.max():.3f}]  mean: {raw_fake.mean():.3f}  std: {raw_fake.std():.3f}"
                            )
                            print(
                                f"  [disc debug] norm_real range: [{real_trans.min():.3f}, {real_trans.max():.3f}]  mean: {real_trans.mean():.3f}  std: {real_trans.std():.3f}"
                            )
                            print(
                                f"  [disc debug] norm_fake range: [{fake_trans.min():.3f}, {fake_trans.max():.3f}]  mean: {fake_trans.mean():.3f}  std: {fake_trans.std():.3f}"
                            )
                            print(
                                f"  [disc debug] real_logits range: [{disc_real_logits.min():.3f}, {disc_real_logits.max():.3f}]  mean: {disc_real_logits.mean():.3f}"
                            )
                            print(
                                f"  [disc debug] fake_logits range: [{disc_fake_logits.min():.3f}, {disc_fake_logits.max():.3f}]  mean: {disc_fake_logits.mean():.3f}"
                            )

        # ==============================================================
        # Logging
        # ==============================================================
        for g in range(G):
            y_pred = b_values_g[:, g].cpu().numpy()
            y_true = b_returns_g[:, g].cpu().numpy()
            var_y = np.var(y_true)
            ev = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            writer.add_scalar(f"losses/explained_variance_g{g}", ev, global_step)

        group_names = ["task", "style"]
        for g in range(G):
            writer.add_scalar(
                f"charts/mean_reward_{group_names[g]}",
                rewards_g[:, :, g].mean().item(),
                global_step,
            )

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "losses/value_loss",
            v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss,
            global_step,
        )
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        if args.lcp_coef > 0:
            writer.add_scalar("losses/grad_penalty", grad_penalty.item(), global_step)

        mean_disc_loss = np.mean(disc_losses) if disc_losses else 0.0
        mean_disc_gp = np.mean(disc_gp_losses) if disc_gp_losses else 0.0
        mean_real_acc = np.mean(disc_real_acc) if disc_real_acc else 0.0
        mean_fake_acc = np.mean(disc_fake_acc) if disc_fake_acc else 0.0
        writer.add_scalar("disc/loss", mean_disc_loss, global_step)
        writer.add_scalar("disc/grad_penalty", mean_disc_gp, global_step)
        writer.add_scalar("disc/real_accuracy", mean_real_acc, global_step)
        writer.add_scalar("disc/fake_accuracy", mean_fake_acc, global_step)
        writer.add_scalar("disc/valid_transitions", n_valid, global_step)

        sps = int(global_step / (time.time() - start_time))
        print(
            f"iter {iteration}/{args.num_iterations} | "
            f"SPS: {sps} | "
            f"task_r: {rewards_g[:,:,0].mean():.3f} | "
            f"style_r: {rewards_g[:,:,1].mean():.3f} | "
            f"disc_loss: {mean_disc_loss:.3f} | "
            f"disc_acc: R={mean_real_acc:.2f} F={mean_fake_acc:.2f} | "
            f"valid_trans: {n_valid}"
        )
        writer.add_scalar("charts/SPS", sps, global_step)

    save_checkpoint()
    envs.close()
    writer.close()
