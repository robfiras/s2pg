from s2pg.environments.point_mass.point_mass_v3 import PointMassPOMDP_v3

from mushroom_rl.environments.gym_env import Gym
Gym.register()
from gym.envs.registration import register
from .gym_mujoco_pomdp import RandomizedMassGym


register(
    id='AntPOMDP-v3',
    entry_point='s2pg.environments.gym_mujoco_pomdp:AntEnvPOMPD',
    max_episode_steps=1000,
)
register(
    id='HalfCheetahPOMDP-v3',
    entry_point='s2pg.environments.gym_mujoco_pomdp:HalfCheetahEnvPOMPD',
    max_episode_steps=1000,
)
register(
    id='HopperPOMDP-v3',
    entry_point='s2pg.environments.gym_mujoco_pomdp:HopperEnvPOMPD',
    max_episode_steps=1000,
)
register(
    id='HumanoidPOMDP-v3',
    entry_point='s2pg.environments.gym_mujoco_pomdp:HumanoidEnvPOMPD',
    max_episode_steps=1000,
)
register(
    id='Walker2dPOMDP-v3',
    entry_point='s2pg.environments.gym_mujoco_pomdp:Walker2dEnvPOMPD',
    max_episode_steps=1000,
)
register(
    id='InvertedPendulumEnvPOMDP',
    entry_point='s2pg.environments.gym_mujoco_pomdp:InvertedPendulumEnvPOMDP',
)
register(
    id='PendulumEnvPOMDP',
    entry_point='s2pg.environments.gym_mujoco_pomdp:PendulumEnvPOMDP',
)


