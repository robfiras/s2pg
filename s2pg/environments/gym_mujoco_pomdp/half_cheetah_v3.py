import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class HalfCheetahEnvPOMPD(HalfCheetahEnv):

    def __init__(self, obs_to_hide=("velocities",), random_force_com=False, max_force_strength=5.0,
                 include_body_vel=False, **kwargs):

        self._hidable_obs = ("positions", "velocities")
        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert not all(x in obs_to_hide for x in self._hidable_obs), "You are not allowed to hide all observations!"
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s."\
                                                                 % (self._hidable_obs,)
        self._obs_to_hide = obs_to_hide
        self._random_force_com = random_force_com
        self._max_force_strength = max_force_strength
        self._force_strength = 0.0
        self._include_body_vel = include_body_vel
        super().__init__(**kwargs)

    def reset_model(self):
        if self._random_force_com:
            self._force_strength = np.random.choice([-self._max_force_strength, self._max_force_strength])
            sign = -1.0 if self._force_strength < 0 else 1.0
            self._forward_reward_weight = np.abs(self._forward_reward_weight) * sign
        return super().reset_model()

    def step(self, action):
        torso_index = self.model.body_names.index('torso')
        self.data.xfrc_applied[torso_index, 0] = self._force_strength
        return super().step(action)

    def _get_obs(self):
        observations = []
        if "positions" not in self._obs_to_hide:
            position = self.sim.data.qpos.flat.copy()
            if self._exclude_current_positions_from_observation:
                position = position[1:]
            observations += [position]

        if "velocities" not in self._obs_to_hide:
            velocity = self.sim.data.qvel.flat.copy()
            observations += [velocity]

        if "velocities" in self._obs_to_hide and self._include_body_vel:
            velocity = self.sim.data.qvel.flat.copy()
            observations += [velocity[:3]]

        return np.concatenate(observations).ravel()

    def get_mask(self, obs_to_hide):
        """ This function returns a boolean mask to hide observations from a fully observable state. """

        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)
        mask = []
        position = self.sim.data.qpos.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        velocity = self.sim.data.qvel.flat.copy()

        if "positions" not in obs_to_hide:
            mask += [np.ones_like(position, dtype=bool)]
        else:
            mask += [np.zeros_like(position, dtype=bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones_like(velocity, dtype=bool)]
        else:
            velocity_mask = [np.zeros_like(velocity, dtype=bool)]
            if self._include_body_vel:
                velocity_mask[0][:3] = 1
            mask += velocity_mask

        return np.concatenate(mask).ravel()
