import gymnasium as gym

from rlink.envs.wrappers.momdp_wrapper import POMDPWrapper


class POMDPMujocoEnv:
    """
    Use this class to register POMDP environments in this file.

    Example:
        from rlink.envs.momdp_mujoco_envs import POMDPMujocoEnv
        gym.register_envs(POMDPMujocoEnv)
    """

    pass


# HalfCheetah-v4
# https://gymnasium.farama.org/environments/mujoco/half_cheetah/#observation-space
def HalfCheetah_P_factory(**kwargs) -> gym.Env:
    env_id = "HalfCheetah-v4"
    visible_obs_dim = [0, 1, 2, 3, 4, 5, 6, 7]
    env = POMDPWrapper(env_id, visible_obs_dim, **kwargs)
    return env


def HalfCheetah_V_factory(**kwargs) -> gym.Env:
    env_id = "HalfCheetah-v4"
    visible_obs_dim = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    env = POMDPWrapper(env_id, visible_obs_dim, **kwargs)
    return env


gym.register(id="HalfCheetahP-v4", entry_point=HalfCheetah_P_factory)
gym.register(id="HalfCheetahV-v4", entry_point=HalfCheetah_V_factory)


# Hopper-v4
# https://gymnasium.farama.org/environments/mujoco/hopper/#observation-space
def Hopper_P_factory(**kwargs) -> gym.Env:
    env_id = "Hopper-v4"
    visible_obs_dim = [0, 1, 2, 3, 4]
    env = POMDPWrapper(env_id, visible_obs_dim, **kwargs)
    return env


def Hopper_V_factory(**kwargs) -> gym.Env:
    env_id = "Hopper-v4"
    visible_obs_dim = [5, 6, 7, 8, 9, 10]
    env = POMDPWrapper(env_id, visible_obs_dim, **kwargs)
    return env


gym.register(id="HopperP-v4", entry_point=Hopper_P_factory)
gym.register(id="HopperV-v4", entry_point=Hopper_V_factory)


# Walker2d-v4
# https://gymnasium.farama.org/environments/mujoco/walker2d/#observation-space
def Walker2D_P_factory(**kwargs) -> gym.Env:
    env_id = "Walker2D-v4"
    visible_obs_dim = [0, 1, 2, 3, 4, 5, 6, 7]
    env = POMDPWrapper(env_id, visible_obs_dim, **kwargs)
    return env


def Walker2D_V_factory(**kwargs) -> gym.Env:
    env_id = "Walker2D-v4"
    visible_obs_dim = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    env = POMDPWrapper(env_id, visible_obs_dim, **kwargs)
    return env


gym.register(id="Walker2DP-v4", entry_point=Walker2D_P_factory)
gym.register(id="Walker2DV-v4", entry_point=Walker2D_V_factory)
