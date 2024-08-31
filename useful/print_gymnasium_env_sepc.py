from pprint import pprint

import gymnasium as gym


def main(env_id: str = "CartPole-v1") -> None:
    env = gym.make(env_id)

    print("===== env.spec =====")
    pprint(env.spec)
    print()

    print("===== env.observation_space =====")
    print(env.observation_space)
    print()

    print("===== env.action_space =====")
    print(env.action_space)
    print()


if __name__ == "__main__":
    env_id = "CartPole-v1"
    main(env_id)
