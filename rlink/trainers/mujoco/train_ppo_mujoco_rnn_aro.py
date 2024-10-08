import collections
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

from rlink.buffers.rollout_buffer_torch import RolloutBufferTorch
from rlink.utils import common, th_util, time_util


@dataclass
class Args:
    project_name: str = "RLink"
    """the wandb's project name and the parent directory for logging"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment, the wandb's group name and the sub-directory for logging"""
    run_name: str | None = None
    """
    the name of this run, the wandb's run name and leaf directory for logging
    (defaults to `exp_name--%Y%m%d_%H%M%S`)
    """
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    use_cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    use_mps: bool = False
    """if toggled, mps will be enabled by default"""
    use_wandb: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    capture_video_eval: bool = False
    """whether to capture videos of the agent performances during evaluation (check out `eval` folder)"""  # noqa: E501
    print_interval: int = 5
    """the interval (iterations) for printing to the console"""
    record_interval: int = 100
    """the interval (episodes) to record videos"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
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
    minibatch_size: int = 64
    """the mini-batch size"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    seq_len: int = 3
    """the sequence length for context"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    output_dir: str = ""
    """the output directory for logging (computed in runtime)"""
    device: str = ""
    """the device (cpu, cuda, mps) used in this experiment (computed in runtime)"""


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    video_path: str,
    record_interval: int,
    gamma: float,
) -> callable:
    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                video_path,
                episode_trigger=lambda x: x % record_interval == 0,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class Agent(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # Critic
        self.critic_oa_encoder = nn.GRU(in_features + out_features, 128, num_layers=1)
        self._critic_prev_a_encoer = nn.GRU(out_features, 32, num_layers=1)
        self._critic_r_encoder = nn.GRU(1, 32, num_layers=1)
        self._critic_o_encoder = nn.GRU(in_features, 64, num_layers=1)
        self.critic_aro_mixer = nn.GRU(128, 128, num_layers=1)
        self.critic = nn.Sequential(
            th_util.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            th_util.layer_init(nn.Linear(256, 1), std=1.0),
        )

        # Actor
        self.actor_o_encoder = nn.GRU(in_features, 128, num_layers=1)
        self._actor_prev_a_encoder = nn.GRU(out_features, 32, num_layers=1)
        self._actor_r_encoder = nn.GRU(1, 32, num_layers=1)
        self._actor_o_encoder = nn.GRU(in_features, 64, num_layers=1)
        self.actor_aro_mixer = nn.GRU(128, 128, num_layers=1)
        self.actor_mean = nn.Sequential(
            th_util.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            th_util.layer_init(nn.Linear(256, out_features), std=0.01),
        )
        self.actor_logstd = nn.Parameter(th.zeros(1, out_features))

        for name, param in self.named_parameters():
            if "encoder" in name or "mixer" in name:
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

    def get_value(
        self,
        obs: th.Tensor,
        prev_action: th.Tensor,
        reward: th.Tensor,
        action: th.Tensor,
    ) -> th.Tensor:
        x = th.cat([obs, action], dim=-1)
        x, _ = self.critic_oa_encoder(x)

        _a, _ = self._critic_prev_a_encoer(prev_action)
        _r, _ = self._critic_r_encoder(reward)
        _o, _ = self._critic_o_encoder(obs)
        aro, _ = self.critic_aro_mixer(th.cat([_a, _r, _o], dim=-1))

        x = th.cat([x[-1], aro[-1]], dim=-1)  # get last hidden state
        value = self.critic(x)
        return value  # (num_envs, 1)

    def get_action(
        self,
        obs: th.Tensor,
        prev_action: th.Tensor,
        reward: th.Tensor,
        action: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_mean = self._get_actor_mean(obs, prev_action, reward)  # (num_envs, out_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (num_envs, out_features)
        action_std = th.exp(action_logstd)  # (num_envs, out_features)
        probs = Normal(action_mean, action_std)  # (num_envs, out_features)
        if action is None:
            action = probs.sample()
        # action: (num_envs, out_features)
        # probs.log_prob(action): (num_envs, out_features)
        # probs.log_prob(action).sum(1, keepdim=True): (num_envs, 1)
        # probs.entropy(): (num_envs, out_features)
        # probs.entropy().sum(1): (num_envs,)
        return action, probs.log_prob(action).sum(1, keepdim=True), probs.entropy().sum(1)

    def _get_actor_mean(
        self,
        obs: th.Tensor,
        prev_action: th.Tensor,
        reward: th.Tensor,
    ) -> th.Tensor:
        x, _ = self.actor_o_encoder(obs)

        _a, _ = self._actor_prev_a_encoder(prev_action)
        _r, _ = self._actor_r_encoder(reward)
        _o, _ = self._actor_o_encoder(obs)
        aro, _ = self.actor_aro_mixer(th.cat([_a, _r, _o], dim=-1))

        x = th.cat([x[-1], aro[-1]], dim=-1)  # get last hidden state
        actor_mean = self.actor_mean(x)
        return actor_mean


def train_ppo(args: Args, Agent: type[Agent]) -> None:
    # Set hyperparameters
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    device = common.get_device(args.use_cuda, args.use_mps)
    args.device = str(device)

    # Set output directory
    if args.run_name is None:
        args.run_name = f"{args.exp_name}--s{args.seed}--{time_util.get_now_str()}"
    args.output_dir = f"runs/{args.project_name}/{args.exp_name}/{args.run_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    if args.use_wandb:
        import wandb

        os.makedirs("runs", exist_ok=True)
        wandb.init(
            project=args.project_name,
            entity=args.wandb_entity,
            group=args.exp_name,
            name=args.run_name,
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
            dir="runs",
        )
    writer = common.create_summary_writer(args.output_dir, vars(args))
    common.save_code_snapshot(args.output_dir)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = args.torch_deterministic

    # Make envs
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                idx=i,
                capture_video=args.capture_video,
                video_path=f"{args.output_dir}/videos",
                record_interval=args.record_interval,
                gamma=args.gamma,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Initialize agent
    agent = Agent(
        in_features=np.array(envs.single_observation_space.shape).prod(),
        out_features=np.array(envs.single_action_space.shape).prod(),
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize buffer
    rollout_buffer = RolloutBufferTorch(
        n_envs=args.num_envs,
        buffer_size=args.num_steps,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
    )
    obs_queue = th_util.TensorQueue(seq_len=args.seq_len)  # for context
    prev_action_queue = th_util.TensorQueue(seq_len=args.seq_len)  # for context
    reward_queue = th_util.TensorQueue(seq_len=args.seq_len)  # for context
    action_queue = th_util.TensorQueue(seq_len=args.seq_len)  # for context

    # Logging variables
    epi_l_queue = collections.deque(maxlen=30)
    epi_r_queue = collections.deque(maxlen=30)

    # Initialize environment
    global_step = 0
    global_episode = 0
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    obs = th.tensor(obs, dtype=th.float32, device=device)  # (num_envs, *obs_shape)
    done = th.zeros((args.num_envs, 1), device=device)  # (num_envs, 1)
    prev_action = th.zeros(
        args.num_envs,
        np.array(envs.single_action_space.shape).prod(),
        device=device,
    )  # (num_envs, *action_shape)
    reward = th.zeros((args.num_envs, 1), device=device)  # (num_envs, 1)
    obs_queue.append(obs)
    prev_action_queue.append(prev_action)
    reward_queue.append(reward)

    for iteration in range(1, args.num_iterations + 1):
        collect_start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_buffer.reset()
        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # Get action
            with th.no_grad():
                obs_seq = obs_queue.get_seq()
                prev_action_seq = prev_action_queue.get_seq()
                reward_seq = reward_queue.get_seq()
                action, logprob, _ = agent.get_action(obs_seq, prev_action_seq, reward_seq)
                action_queue.append(action)
                action_seq = action_queue.get_seq()
                value = agent.get_value(
                    obs_seq, prev_action_seq, reward_seq, action_seq
                )  # (num_envs, 1)

            # env.step
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # Postprocess data
            next_obs = th.tensor(next_obs, dtype=th.float32, device=device)
            next_done = np.logical_or(terminations, truncations)
            next_done = th.tensor(next_done, dtype=th.float32, device=device).unsqueeze(1)
            reward = th.tensor(reward, dtype=th.float32, device=device).unsqueeze(1)
            # next_obs: (num_envs, *obs_shape)
            # next_done: (num_envs, 1)
            # reward: (num_envs, 1)

            # queue for context
            prev_action_queue.append(action)
            reward_queue.append(reward)

            # bootstrap value if truncated
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L213
            for idx, truncation in enumerate(truncations):
                if truncation:
                    final_prev_action_seq = prev_action_queue.get_single_seq(idx)
                    final_obs = infos["final_observation"]  # (num_envs, *obs_shape)
                    final_obs = final_obs[idx].reshape(1, -1)  # (1, *obs_shape)
                    final_obs = th.tensor(final_obs, dtype=th.float32, device=device)
                    final_obs_seq = obs_queue.get_single_seq(idx, final_obs)
                    final_reward_seq = reward_queue.get_single_seq(idx)
                    # final_prev_action_seq: (seq_len, 1, *action_shape)
                    # final_obs_seq: (seq_len, 1, *obs_shape)
                    # final_reward_seq: (seq_len, 1, 1)
                    with th.no_grad():
                        final_action, _, _ = agent.get_action(
                            obs=final_obs_seq,
                            prev_action=final_prev_action_seq,
                            reward=final_reward_seq,
                        )  # (1, *action_shape)
                        final_action_seq = action_queue.get_single_seq(idx, final_action)
                        # final_action_seq: (seq_len, 1, *action_shape)
                        final_value = agent.get_value(
                            obs=final_obs_seq,
                            prev_action=final_prev_action_seq,
                            reward=final_reward_seq,
                            action=final_action_seq,
                        )  # (1, 1)
                    reward[idx] += args.gamma * final_value.squeeze()  # scalar

            # Add transition to buffer
            rollout_buffer.add(obs, action, reward, done, logprob, value)

            # Update data for next step
            obs = next_obs
            done = next_done
            prev_action = action
            obs_queue.append(obs)
            global_episode += int(th.sum(done).item())

            # Logging a episode for one env at the same time
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        epi_r = info["episode"]["r"]
                        epi_l = info["episode"]["l"]
                        epi_l_queue.append(epi_l)
                        epi_r_queue.append(epi_r)
                        writer.add_scalar("Charts/episodic_return", epi_r, global_step)
                        writer.add_scalar("Charts/episodic_length", epi_l, global_step)
                    break

        collect_time = time.time() - collect_start_time
        train_start_time = time.time()

        # Compute value for the last step after the num_steps
        # bootstrap value if not done
        with th.no_grad():
            next_obs_seq = obs_queue.get_seq()
            action_seq = prev_action_queue.get_seq()
            reward_seq = reward_queue.get_seq()
            next_action, _, _ = agent.get_action(
                next_obs_seq, action_seq, reward_seq
            )  # (num_envs, *action_shape)
            next_action_seq = action_queue.get_seq(next_action)
            next_value = agent.get_value(
                next_obs_seq, action_seq, reward_seq, next_action_seq
            )  # (num_envs, 1)
            rollout_buffer.compute_returns_and_advantages(next_value, next_done)

        # Update policy
        train_data = train_step(
            args=args,
            agent=agent,
            b_obs=rollout_buffer.get("b_obs"),
            b_action=rollout_buffer.get("b_action"),
            b_log_prob=rollout_buffer.get("b_log_prob"),
            b_value=rollout_buffer.get("b_value"),
            b_advantage=rollout_buffer.get("b_advantage"),
            b_return=rollout_buffer.get("b_return"),
            b_reward=rollout_buffer.get("b_reward"),
            optimizer=optimizer,
        )

        train_time = time.time() - train_start_time

        v_loss = train_data["v_loss"]
        pg_loss = train_data["pg_loss"]
        entropy_loss = train_data["entropy_loss"]
        old_approx_kl = train_data["old_approx_kl"]
        approx_kl = train_data["approx_kl"]
        clip_frac = train_data["clip_frac"]
        explained_var = train_data["explained_variance"]

        # Logging - tensorboard
        sps = int(global_step / (time.time() - start_time))
        etc = time_util.get_etc(args.total_timesteps, global_step, start_time, time.time())
        etc_str = time_util.time_to_str(etc)
        writer.add_scalar("Charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("Losses/value_loss", v_loss, global_step)
        writer.add_scalar("Losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("Losses/entropy", entropy_loss, global_step)
        writer.add_scalar("Losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("Losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("Losses/clipfrac", clip_frac, global_step)
        writer.add_scalar("Losses/explained_variance", explained_var, global_step)
        writer.add_scalar("Charts/sps", sps, global_step)
        writer.add_scalar("Charts/elapsed_time", time.time() - start_time, global_step)
        writer.add_scalar("Charts/global_episode", global_episode, global_step)
        writer.add_scalar("Charts/collection_time", collect_time, global_step)
        writer.add_scalar("Charts/train_time", train_time, global_step)
        # Logging - console
        if iteration % args.print_interval == 0:
            print(
                f"Iter: {iteration}/{args.num_iterations}  "
                f"Epi R: {np.mean(epi_r_queue):5.1f}  "
                f"Epi L: {np.mean(epi_l_queue):.0f}  "
                f"sps: {sps:4.0f}  "
                f"etc: {etc_str}  "
                f"Value Loss: {v_loss:.3f}  "
                f"Policy Loss: {pg_loss:.3f}  "
                f"Entropy Loss: {entropy_loss:.3f}  "
            )

    if args.save_model:
        model_path = f"{args.output_dir}/latest_model.pt"
        th.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from rlink.evaluators.mujoco.eval_ppo_mujoco_rnn_aro import ppo_evaluate

        episodic_returns = ppo_evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            video_path=f"{args.output_dir}/eval",
            Model=Agent,
            model_kwargs={
                "in_features": np.array(envs.single_observation_space.shape).prod(),
                "out_features": np.array(envs.single_action_space.shape).prod(),
            },
            seq_len=args.seq_len,
            device=device,
            capture_video=args.capture_video_eval,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
    if args.use_wandb:
        wandb.finish()


def train_step(
    args: Args,
    agent: Agent,
    b_obs: th.Tensor,
    b_action: th.Tensor,
    b_log_prob: th.Tensor,
    b_value: th.Tensor,
    b_advantage: th.Tensor,
    b_return: th.Tensor,
    b_reward: th.Tensor,
    optimizer: optim.Optimizer,
) -> dict[str, float]:
    b_inds = np.arange(args.batch_size)  # batch_size = num_steps * num_envsps
    clip_fracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]  # (mb_size,)

            # for context
            seq_offsets = np.arange(-args.seq_len + 1, 1).reshape(-1, 1)  # (seq_len, 1)
            seq_offsets *= args.num_envs  # (seq_len, 1),  considering env dim
            _mb_inds = mb_inds.reshape(1, -1)  # (1, mb_size)
            seq_mb_inds = _mb_inds + seq_offsets  # (seq_len, mb_size)
            padding_mask = seq_mb_inds >= 0  # True: valid index, False: padding index
            seq_mb_inds = seq_mb_inds.clip(min=0)  # clip negative index to 0

            # Sample mini-batch
            mb_obs = b_obs[seq_mb_inds]  # (seq_len, mb_size, *obs_shape)
            mb_obs[~padding_mask] = 0  # padding with 0
            mb_action = b_action[seq_mb_inds]  # (seq_len, mb_size, *action_shape)
            mb_action[~padding_mask] = 0  # padding with 0
            mb_log_prob = b_log_prob[mb_inds]  # (mb_size, 1)
            mb_value = b_value[mb_inds]  # (mb_size, 1)
            mb_advantage = b_advantage[mb_inds]  # (mb_size, 1)
            mb_return = b_return[mb_inds]  # (mb_size, 1)
            mb_reward = b_reward[seq_mb_inds]  # (seq_len, mb_size, 1)
            mb_reward[~padding_mask] = 0  # padding with 0

            # for previous action
            prev_seq_mb_inds = seq_mb_inds - args.num_envs  # (mb_size,)
            mask_prev = prev_seq_mb_inds >= 0  # True: valid index, False: padding index
            prev_seq_mb_inds = prev_seq_mb_inds.clip(min=0)  # clip negative index to 0
            mb_prev_action = b_action[prev_seq_mb_inds]  # (seq_len, mb_size, *action_shape)
            mb_prev_action[~mask_prev] = 0  # padding with 0

            _, new_log_prob, entropy = agent.get_action(
                obs=mb_obs,
                prev_action=mb_prev_action,
                reward=mb_reward,
                action=mb_action,
            )
            # new_log_prob: (mb_size, 1)
            # entropy: (mb_size,)
            new_value = agent.get_value(
                mb_obs, mb_prev_action, mb_reward, mb_action
            )  # (mb_size, 1)
            log_ratio = new_log_prob - mb_log_prob  # (mb_size, 1)
            ratio = log_ratio.exp()  # (mb_size, 1)

            with th.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-log_ratio).mean()  # (mb_size, 1) -> scalar
                approx_kl = ((ratio - 1) - log_ratio).mean()  # (mb_size, 1) -> scalar
                clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            if args.norm_adv:
                mb_advantage = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std() + 1e-8)

            # Policy loss
            clipped_ratio = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)  # (mb_size, 1)
            pg_loss1 = -mb_advantage * ratio  # (mb_size, 1)
            pg_loss2 = -mb_advantage * clipped_ratio  # (mb_size, 1)
            pg_loss = th.max(pg_loss1, pg_loss2).mean()  # (mb_size, 1) -> scalar

            # Value loss
            if args.clip_vloss:
                v_loss_unclipped = (new_value - mb_return) ** 2  # (mb_size, 1)
                v_clipped = mb_value + th.clamp(  # (mb_size, 1)
                    new_value - mb_value,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_return) ** 2  # (mb_size, 1)
                v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)  # (mb_size, 1)
                v_loss = 0.5 * v_loss_max.mean()  # scalar
            else:
                v_loss = 0.5 * ((new_value - mb_return) ** 2).mean()  # (mb_size, 1) -> scalar

            entropy_loss = entropy.mean()  # (mb_size, 1) -> scalar
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  # scalar

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred = b_value.cpu().numpy()
    y_true = b_return.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "v_loss": v_loss.item(),
        "pg_loss": pg_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clip_frac": np.mean(clip_fracs),
        "explained_variance": explained_var,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    train_ppo(args, Agent)
