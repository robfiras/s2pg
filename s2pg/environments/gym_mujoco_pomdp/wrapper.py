import numpy as np
from copy import deepcopy
from mushroom_rl.environments import Gym

from gym.spaces import Box


class RandomizedMassGym(Gym):

    def __init__(self, name, random_action_rewiring=False, **kwargs):
        super(RandomizedMassGym, self).__init__(name, **kwargs)

        self._allowed_relative_mass_change = [-0.3, 0.0, 0.3]
        self._init_geom_size = deepcopy(self.env.model.geom_size)
        self._init_body_mass = deepcopy(self.env.model.body_mass)
        self._curr_mass_change = np.zeros_like(self._init_body_mass)
        self._random_action_rewiring = random_action_rewiring
        if random_action_rewiring:
            self._action_wiring_map = np.arange(self.env.action_space.shape[0])
            np.random.shuffle(self._action_wiring_map)
        else:
            self._action_wiring_map = np.arange(self.env.action_space.shape[0])

        # update observation space of env
        observation_space = self.env.observation_space
        if random_action_rewiring:
            low = np.concatenate([observation_space.low,
                                  np.ones_like(self._init_body_mass) * np.min(self._allowed_relative_mass_change),
                                  np.ones_like(self._action_wiring_map) * np.min(self._action_wiring_map)])
            high = np.concatenate([observation_space.high,
                                  np.ones_like(self._init_body_mass) * np.max(self._allowed_relative_mass_change),
                                  np.ones_like(self._action_wiring_map) * np.max(self._action_wiring_map)])
        else:
            low = np.concatenate([observation_space.low,
                                  np.ones_like(self._init_body_mass) * np.min(self._allowed_relative_mass_change)])
            high = np.concatenate([observation_space.high,
                                  np.ones_like(self._init_body_mass) * np.max(self._allowed_relative_mass_change)])
        new_observation_space = Box(low, high, dtype=observation_space.dtype)
        self.env.observation_space = new_observation_space
        self.info.observation_space = new_observation_space
        self.env.env.observation_space = new_observation_space

    def reset(self, state=None):

        obs = super(RandomizedMassGym, self).reset(state)
        body_geom_adr = deepcopy(self.env.model.body_geomadr)
        self._curr_mass_change = np.random.choice(self._allowed_relative_mass_change, len(self._init_body_mass))
        mult =  1 + self._curr_mass_change
        for i in range(len(self.env.model.body_mass)):
            if i < len(self.env.model.body_mass)-1:
                start = body_geom_adr[i]
                end = body_geom_adr[i+1]
                self.env.model.geom_size[start:end, 0] = self._init_geom_size[start:end, 0] * mult[i]
            else:
                start = body_geom_adr[i]
                self.env.model.geom_size[start:, 0] = self._init_geom_size[start:, 0] * mult[i]
        self.env.model.body_mass[:] = self._init_body_mass[:] * (1 + self._curr_mass_change)

        # add mass info
        obs = np.concatenate([obs, self._curr_mass_change])

        # sample action rewiring map
        if self._random_action_rewiring:
            np.random.shuffle(self._action_wiring_map)
            obs = np.concatenate([obs, self._action_wiring_map])

        return obs

    def step(self, action):

        action = action[self._action_wiring_map]

        obs, reward, absorbing, info = super(RandomizedMassGym, self).step(action)

        # add mass info
        obs = np.concatenate([obs, self._curr_mass_change])

        if self._random_action_rewiring:
            obs = np.concatenate([obs, self._action_wiring_map])

        return obs, reward, absorbing, info

    def get_mask(self, obs_to_hide, hide_mass_setup=True):

        mask = self.env.get_mask(obs_to_hide)
        if hide_mass_setup:
            mask = np.concatenate([mask, np.zeros_like(self._curr_mass_change, dtype=bool)])
        else:
            mask = np.concatenate([mask, np.ones_like(self._curr_mass_change, dtype=bool)])

        if self._random_action_rewiring:
            mask = np.concatenate([mask, np.zeros_like(self._action_wiring_map, dtype=bool)])

        return mask


if __name__ == "__main__":
    mdp = RandomizedMassGym("HalfCheetahPOMDP-v3", horizon=1000)
    mask = mdp.get_mask(("positions",))

    for i in range(100):
        mdp.reset()
        for j in range(10):
            a = np.random.randn(mdp.env.action_space.shape[0])
            state, reward, absorbing, info = mdp.step(a)
            mdp.render()
