import os
import sys

import tyro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_ppo_mujoco_gru() -> None:
    from rlink.trainers.mujoco.train_ppo_mujoco_rnn import ActorCriticGruPolicy, Args, train_ppo

    args = tyro.cli(Args)
    train_ppo(args, ActorCriticGruPolicy)


if __name__ == "__main__":
    train_ppo_mujoco_gru()
