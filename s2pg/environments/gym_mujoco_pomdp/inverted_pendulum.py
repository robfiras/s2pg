import numpy as np
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv


class InvertedPendulumEnvPOMDP(InvertedPendulumEnv):

    def __init__(self, obs_to_hide=("velocities",), **kwargs):
        self._hidable_obs = ("positions", "velocities")
        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert not all(x in obs_to_hide for x in self._hidable_obs), "You are not allowed to hide all observations!"
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)
        self._obs_to_hide = obs_to_hide

        super().__init__(**kwargs)

    def _get_obs(self):
        observations = []
        if "positions" not in self._obs_to_hide:
            position = self.sim.data.qpos.flat.copy()
            observations += [position]

        if "velocities" not in self._obs_to_hide:
            velocity = self.sim.data.qvel.flat.copy()
            observations += [velocity]

        return np.concatenate(observations).ravel()
