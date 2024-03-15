import numpy as np
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv


class HumanoidEnvPOMPD(HumanoidEnv):

    def __init__(self, obs_to_hide=("velocities", "com_inertia", "com_velocity", "actuator_forces"),
                 include_body_vel=False, random_force_com=False, max_force_strength=10.0, **kwargs):

        self._hidable_obs = ("positions", "velocities", "com_inertia", "com_velocity",
                             "actuator_forces", "external_contact_forces")
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
                position = position[2:]
            observations += [position]

        if "velocities" not in self._obs_to_hide:
            velocity = self.sim.data.qvel.flat.copy()
            observations += [velocity]

        if "com_inertia" not in self._obs_to_hide:
            com_inertia = self.sim.data.cinert.flat.copy()
            observations += [com_inertia]

        if "com_velocity" not in self._obs_to_hide:
            com_velocity = self.sim.data.cvel.flat.copy()
            observations += [com_velocity]

        if "actuator_forces" not in self._obs_to_hide:
            actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
            observations += [actuator_forces]

        if "external_contact_forces" not in self._obs_to_hide:
            external_contact_forces = self.sim.data.cfrc_ext.flat.copy()
            observations += [external_contact_forces]

        return np.concatenate(observations)

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
            position = position[2:]
        velocity = self.sim.data.qvel.flat.copy()
        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if "positions" not in obs_to_hide:
            mask += [np.ones_like(position, dtype=bool)]
        else:
            mask += [np.zeros_like(position, dtype=bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones_like(velocity, dtype=bool)]
        else:
            velocity_mask = [np.zeros_like(velocity, dtype=bool)]
            if self._include_body_vel:
                velocity_mask[0][:6] = 1
            mask += velocity_mask

        if "com_inertia" not in obs_to_hide:
            mask += [np.ones_like(com_inertia, dtype=bool)]
        else:
            mask += [np.zeros_like(com_inertia, dtype=bool)]

        if "com_velocity" not in obs_to_hide:
            mask += [np.ones_like(com_velocity, dtype=bool)]
        else:
            mask += [np.zeros_like(com_velocity, dtype=bool)]

        if "actuator_forces" not in obs_to_hide:
            mask += [np.ones_like(actuator_forces, dtype=bool)]
        else:
            mask += [np.zeros_like(actuator_forces, dtype=bool)]

        if "external_contact_forces" not in obs_to_hide:
            mask += [np.ones_like(external_contact_forces, dtype=bool)]
        else:
            mask += [np.zeros_like(external_contact_forces, dtype=bool)]

        return np.concatenate(mask).ravel()
