import time

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.dataset import parse_dataset, compute_episodes_length
from mushroom_rl.utils.viewer import Viewer


class PointMassDoor(Environment):

    def __init__(self, horizon=1000, gamma=0.99, n_steps_action=3, observable_radius=20.0,
                 door_width=40.0, ff_render_factor=1.0):
        """
        Constructor.


        Args:
             horizon (int, 1000): horizon of the task
             gamma (float, 0.99): discount factor
             n_steps_action (int, 3): number of integration intervals for each
                                      step of the mdp.

        """
        # MDP parameters
        self.field_width = 150
        self.field_height = 150

        self.goal_pos = np.array([75.0, 140.0])
        self._starting_pos = np.array([75.0, 10.0])
        self._door_range = [10, 100]
        self._door_pos1 = None
        self._door_pos1_height = 60.0
        self._door_pos2 = None
        self._door_pos2_height = 90.0
        self._door_width = door_width
        self._goal_radius = 5.0
        self._observable_radius = observable_radius
        self._point_mass_radius = 2.0

        self._v = 3.
        self._dt = .2

        self._out_reward = -2

        self._last_starting_pos = None
        self._state = self.reset()
        self.n_steps_action = n_steps_action

        # MDP properties
        low = np.array([0, 0, -1, -1])
        high = np.array([self.field_width, self.field_height, self._door_range[1], self._door_range[1]])
        observation_space = spaces.Box(low=low, high=high)
        self._action_max = np.ones(1)
        action_space = spaces.Box(low=-self._action_max, high=self._action_max)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = Viewer(self.field_width, self.field_height, self.field_width*3, self.field_height*3,
                              background=(66, 131, 237))

        self._ff_render_factor = ff_render_factor
        self._iter = 0

        super().__init__(mdp_info)

    def reset(self, state=None):

        # sample random door pos
        self._door_pos1 = np.random.uniform(*self._door_range)
        self._door_pos2 = np.random.uniform(*self._door_range)

        new_state = self._starting_pos
        self._last_starting_pos = new_state

        self._state = np.concatenate([new_state, [self._door_pos1, self._door_pos1,
                                                  self._door_pos2, self._door_pos2]]) # the goal is added twice, once for the
                                                                                      # policy and once for the critic
        self._iter = 0
        return self._state

    def step(self, action):

        # bound action
        action = np.atleast_1d(action)
        ac = self._bound(action[0], -self._action_max, self._action_max)

        # convert to direction
        phi = (0.5*(ac + 1.0))
        phi = phi * 2 * np.pi

        new_state = self._state[:2]

        for _ in range(self.n_steps_action):
            state = new_state
            new_state = np.empty(2)
            new_state[0] = state[0] + np.cos(phi) * self._v * self._dt
            new_state[1] = state[1] + np.sin(phi) * self._v * self._dt


            # clip the state to be withing the field
            new_state = np.clip(new_state, np.array([0.0, 0.0]), np.array([self.field_width, self.field_height]))

            # did the agent leave the field?
            # if new_state[0] > self.field_width \
            #    or new_state[1] > self.field_height \
            #    or new_state[0] < 0 or new_state[1] < 0:
            #
            #     new_state[0] = self._bound(new_state[0], 0, self.field_width)
            #     new_state[1] = self._bound(new_state[1], 0, self.field_height)
            #
            #     reward = self._out_reward
            #     absorbing = True
            #     break
            # did the agent touch the wall?
            if self._door_pos1_height - 2.5 < state[1] < self._door_pos1_height + 2.5 and \
                    not(self._door_pos1+self._point_mass_radius < state[0] < self._door_pos1 + self._door_width-self._point_mass_radius):
                reward = self._out_reward
                absorbing = True
                break
            elif self._door_pos2_height - 2.5 < state[1] < self._door_pos2_height + 2.5 and \
                     not(self._door_pos2+self._point_mass_radius < state[0] < self._door_pos2 + self._door_width-self._point_mass_radius):
                reward = self._out_reward
                absorbing = True
                break
            else:
                reward = 10*np.exp(-0.03 * np.linalg.norm(self.goal_pos - new_state))
                absorbing = False

        if self.in_observable_area(new_state):  # if True, add the goal to the state of the policy
            self._state = np.concatenate([new_state, [self._door_pos1, self._door_pos1,
                                                      self._door_pos2, self._door_pos2 ]])
        else:
            self._state = np.concatenate([new_state, [-1.0, self._door_pos1, -1.0, self._door_pos2]])

        self._iter += 1

        return self._state, reward, absorbing, {}

    def render(self, mode='human'):

        # observable area
        self._viewer.circle(self._last_starting_pos, self._observable_radius, color=(86, 151, 255))

        # goal
        if self.in_goal(self._state):
            self._viewer.circle(self.goal_pos, self._goal_radius, color=(124, 252, 0))
        elif not self.in_observable_area(self._state):
            self._viewer.circle(self.goal_pos, self._goal_radius, color=(150, 150, 150))
        else:
            self._viewer.circle(self.goal_pos, self._goal_radius, color=(210, 43, 43))

        # point mass
        self._viewer.circle(self._state, self._point_mass_radius, color=(0, 0, 0))

        # door1
        self._viewer.line(np.array([0.0, self._door_pos1_height]), np.array([self._door_pos1, self._door_pos1_height]),
                          width=10, color=(0, 0, 0))
        self._viewer.line(np.array([self._door_pos1 + self._door_width, self._door_pos1_height]),
                          np.array([150.0, self._door_pos1_height]), width=10, color=(0, 0, 0))

        # door2
        self._viewer.line(np.array([0.0, self._door_pos2_height]), np.array([self._door_pos2, self._door_pos2_height]),
                          width=10, color=(0, 0, 0))
        self._viewer.line(np.array([self._door_pos2 + self._door_width, self._door_pos2_height]),
                          np.array([150.0, self._door_pos2_height]), width=10, color=(0, 0, 0))


        self._viewer.display(self._dt/self._ff_render_factor)

    def get_viewer(self):
        return self._viewer

    def stop(self):
        self._viewer.close()

    def critic_state_mask(self):
        return np.concatenate([np.ones_like(self._state[:2], dtype=bool),
                               [False, True, False, True]])

    def policy_state_mask(self):
        return np.concatenate([np.ones_like(self._state[:2], dtype=bool),
                               [True, False, True, False]])

    def in_goal(self, state):
        return self.goal_pos[0] - self._goal_radius <= state[0] <= self.goal_pos[0] + self._goal_radius \
               and self.goal_pos[1] - self._goal_radius <= state[1] <= self.goal_pos[1] + self._goal_radius

    def in_observable_area(self, state):
        dist_from_start = np.linalg.norm(state[:2] - self._last_starting_pos)
        return dist_from_start <= self._observable_radius

    def calculate_success_rate(self, dataset, n_episodes):
        state, action, reward, next_state, absorbing, last = parse_dataset(dataset)
        last_ind = np.squeeze(np.argwhere(last == 1))
        state_set = np.split(state, last_ind)
        successes = []
        for state_episode in state_set:
            goal_pos = state_episode[:, 4:6]
            point_mass = state_episode[:, :2]
            dist = np.linalg.norm(goal_pos - point_mass, axis=1)
            success = np.where(dist <= self._goal_radius, 1, 0)
            success = np.minimum(1, np.sum(success))
            successes.append(success)
        sum_successes = np.sum(successes)
        return sum_successes / n_episodes


if __name__=="__main__":

    mdp = PointMassDoor()
    for j in range(100):
        state = mdp.reset()
        print("State: ", state)
        for i in range(200):
            a = -0.5
            state,_, a,_  = mdp.step([a])
            print("State: ", state)
            mdp.render()
            if a:
                break
