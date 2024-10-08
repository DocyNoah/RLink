import os
import sys

import tyro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_ppo_mujoco() -> None:
    from rlink.trainers.mujoco.train_ppo_mujoco_mlp import ActorCriticMlpPolicy, Args, train_ppo

    args = tyro.cli(Args)
    train_ppo(args, ActorCriticMlpPolicy)


if __name__ == "__main__":
    train_ppo_mujoco()
