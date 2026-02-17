# CleanRL PPO with multi-critic per-group reward normalization.
# Based on: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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
from beastlab.client import BeastClientEnv
from beastlab.env_loader import make_beast_gym


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
    """gym env id, or Beast .so module name (e.g. HumanoidEnv)"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.7
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    lcp_coef: float = 1.0
    """coefficient for Lipschitz constraint penalty (gradient penalty on actor)"""

    # Multi-critic
    num_reward_groups: int = 3
    """number of independent reward groups (critics)"""
    reward_weights: str = "2.0,1.0,0.5"
    """comma-separated weights for combining per-group normalized advantages"""

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
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # No NormalizeReward/TransformReward — per-group advantage normalization replaces them
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NormalizedActor(nn.Module):
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
    def __init__(self, envs, num_reward_groups=3):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.num_reward_groups = num_reward_groups

        # One critic per reward group
        self.critics = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            for _ in range(num_reward_groups)
        ])

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_values(self, x):
        """Returns (B, G) tensor of per-group values."""
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
            self.get_values(x),  # (B, G)
        )


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Parse reward weights
    reward_weights = [float(w) for w in args.reward_weights.split(",")]
    G = args.num_reward_groups
    assert len(reward_weights) == G, f"reward_weights length {len(reward_weights)} != num_reward_groups {G}"

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

    # Environment setup
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

    agent = Agent(envs, num_reward_groups=G).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Save function — called on normal exit, Ctrl+C, or SIGTERM
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

        # Extract obs normalization stats
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

    # Storage
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_g = torch.zeros((args.num_steps, args.num_envs, G)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_g = torch.zeros((args.num_steps, args.num_envs, G)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, vals_g = agent.get_action_and_values(next_obs)
                values_g[step] = vals_g  # (E, G)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Extract per-group rewards from info
            # SyncVectorEnv stacks per-env info values; reward_groups may be
            # an object-dtype array of per-env arrays, a 2D float array, or a 1D array (single env).
            if "reward_groups" in infos:
                rg_raw = infos["reward_groups"]
                for ei in range(args.num_envs):
                    if isinstance(rg_raw, np.ndarray) and rg_raw.dtype == object:
                        rg = np.asarray(rg_raw[ei], dtype=np.float32)
                    elif isinstance(rg_raw, np.ndarray) and rg_raw.ndim == 2:
                        rg = rg_raw[ei]
                    elif isinstance(rg_raw, np.ndarray) and rg_raw.ndim == 1:
                        rg = rg_raw
                    else:
                        rg = np.asarray(rg_raw, dtype=np.float32)
                    rewards_g[step, ei] = torch.from_numpy(rg[:G].astype(np.float32)).to(device)
            else:
                for ei in range(args.num_envs):
                    rewards_g[step, ei, 0] = reward[ei] if hasattr(reward, '__getitem__') else reward

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for ei, info in enumerate(infos["final_info"]):
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

        # Per-group GAE
        with torch.no_grad():
            next_values_g = agent.get_values(next_obs)  # (E, G)
            advantages_g = torch.zeros_like(rewards_g).to(device)  # (T, E, G)

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
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )

            returns_g = advantages_g + values_g  # (T, E, G)

            # Combine: normalize each group, weighted sum
            combined_advantages = torch.zeros(args.num_steps, args.num_envs, device=device)
            for g in range(G):
                ag = advantages_g[:, :, g]
                ag_norm = (ag - ag.mean()) / (ag.std() + 1e-8)
                combined_advantages += reward_weights_t[g] * ag_norm

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = combined_advantages.reshape(-1)
        b_returns_g = returns_g.reshape(-1, G)
        b_values_g = values_g.reshape(-1, G)

        # Optimize policy and value network
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

                # Advantages are already per-group normalized, skip norm_adv
                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: sum of per-group MSE losses
                v_loss = 0.0
                for g in range(G):
                    nv = newvalues_g[:, g]
                    if args.clip_vloss:
                        v_loss_unclipped = (nv - b_returns_g[mb_inds, g]) ** 2
                        v_clipped = b_values_g[mb_inds, g] + torch.clamp(
                            nv - b_values_g[mb_inds, g], -args.clip_coef, args.clip_coef
                        )
                        v_loss += (
                            0.5
                            * torch.max(
                                v_loss_unclipped, (v_clipped - b_returns_g[mb_inds, g]) ** 2
                            ).mean()
                        )
                    else:
                        v_loss += 0.5 * ((nv - b_returns_g[mb_inds, g]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # LCP: Lipschitz constraint penalty on actor
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

        # Logging
        # Per-group value explained variance
        for g in range(G):
            y_pred = b_values_g[:, g].cpu().numpy()
            y_true = b_returns_g[:, g].cpu().numpy()
            var_y = np.var(y_true)
            ev = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            writer.add_scalar(f"losses/explained_variance_g{g}", ev, global_step)
            writer.add_scalar(
                f"charts/mean_reward_g{g}",
                rewards_g[:, :, g].mean().item(),
                global_step,
            )

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        if args.lcp_coef > 0:
            writer.add_scalar("losses/grad_penalty", grad_penalty.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    save_checkpoint()
    envs.close()
    writer.close()
