import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv
from gym import spaces


class PendulumEnvPOMDP(PendulumEnv):

    def __init__(self, obs_to_hide=("velocities",), **kwargs):

        super().__init__(**kwargs)

        self._obs_to_hide = obs_to_hide

        high = []
        if "positions" not in self._obs_to_hide:
            high.append([1.0, 1.0])

        if "velocities" not in self._obs_to_hide:
            high.append([self.max_speed])

        high = np.concatenate(high, dtype=np.float32).ravel()
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _get_obs(self):
        theta, thetadot = self.state
        observations = []

        if "positions" not in self._obs_to_hide:
            observations.append([np.cos(theta), np.sin(theta)])

        if "velocities" not in self._obs_to_hide:
            observations.append([thetadot])

        return np.concatenate(observations).ravel()
