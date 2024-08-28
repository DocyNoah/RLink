import gymnasium as gym
import torch as th

from rlink.utils import th_util


def ppo_evaluate(
    model_path: str,
    make_env: callable,
    env_id: str,
    eval_episodes: int,
    video_path: str,
    Model: th.nn.Module,
    model_kwargs: dict,
    seq_len: int,
    device: th.device = th.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
) -> list[float]:
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=env_id,
                idx=0,
                capture_video=capture_video,
                video_path=video_path,
                record_interval=1,
                gamma=gamma,
            )
        ]
    )
    agent = Model(**model_kwargs).to(device)
    agent.load_state_dict(th.load(model_path, map_location=device, weights_only=True))
    agent.eval()

    obs_queue = th_util.SequenceQueue(seq_len)
    obs, _ = envs.reset()
    obs = th.tensor(obs, dtype=th.float32, device=device)
    obs_queue.append(obs)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _ = agent.get_action(obs_queue.get_obs_seq())
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns
