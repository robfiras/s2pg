import time

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.dataset import parse_dataset, compute_episodes_length
from mushroom_rl.utils.viewer import Viewer


class PointMassPOMDP(Environment):

    def __init__(self, horizon=1000, gamma=0.99, n_steps_action=3, ff_render_factor=1.0,
                 goal_area_radius=30.0):
        """
        Constructor.

        Args:
             horizon (int, 1000): horizon of the task
             gamma (float, 0.99): discount factor
             n_steps_action (int, 3): number of integration intervals for each
                                      step of the mdp.

        """
        # MDP parameters
        self.field_size = 150
        low = np.array([0, 0, 0, 0])
        high = np.array([self.field_size, self.field_size, self.field_size, self.field_size])

        self.goal_pos = np.array([75.0, 75.0])
        self._potential_goal_pos = np.array([[75.0, 75.0], [25.0, 25.0], [25.0, 75.0], [100.0, 15.0], [55.0, 108.0]])
        self._goal_radius = 5.0
        self._goal_area_radius = goal_area_radius

        self._v = 3.
        self._dt = .2

        self._goal_reward = 50
        self._out_reward = -2
        self._transition_reward = -0.1

        self._state = self.reset()
        self.n_steps_action = n_steps_action

        # MDP properties
        observation_space = spaces.Box(low=low, high=high)
        self._action_max = np.ones(1)
        action_space = spaces.Box(low=-self._action_max, high=self._action_max)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = Viewer(self.field_size, self.field_size,
                              background=(66, 131, 237))
        self._ff_render_factor = ff_render_factor
        self._iter = 0

        super().__init__(mdp_info)

    def reset(self, state=None):
        # sample random goal
        #self.goal_pos = np.random.rand(2) * 150
        new_goal_ind = np.random.randint(len(self._potential_goal_pos))
        self.goal_pos = self._potential_goal_pos[new_goal_ind]
        if state is None:
            while True:
                new_state = np.random.rand(2)*130 + 10
                if np.linalg.norm(self.goal_pos - new_state) >= self._goal_area_radius + 10:
                    break
        else:
            new_state = state

        self._state = np.concatenate([new_state, self.goal_pos, self.goal_pos]) # the goal is added twice, once for the
                                                                                # policy and once for the critic

        self._iter = 0
        return self._state

    def step(self, action):

        # bound action
        action = np.atleast_1d(action)
        ac = self._bound(action[0], -self._action_max, self._action_max )

        # convert to direction
        phi = (0.5*(ac + 1.0))
        phi = phi * 2 * np.pi

        new_state = self._state[:2]

        for _ in range(self.n_steps_action):
            state = new_state
            new_state = np.empty(2)
            new_state[0] = state[0] + np.cos(phi) * self._v * self._dt
            new_state[1] = state[1] + np.sin(phi) * self._v * self._dt

            if new_state[0] > self.field_size \
               or new_state[1] > self.field_size \
               or new_state[0] < 0 or new_state[1] < 0:

                new_state[0] = self._bound(new_state[0], 0, self.field_size)
                new_state[1] = self._bound(new_state[1], 0, self.field_size)

                reward = self._out_reward
                absorbing = True
                break
            else:
                reward = np.exp(-0.1 * np.linalg.norm(self.goal_pos - new_state))
                absorbing = False
            # elif self.in_goal(state):
            #     reward = self._goal_reward
            #     absorbing = True
            # else:
            #     reward = self._transition_reward
            #     absorbing = False


        if self.in_goal_area(new_state):  # if True, hides the goal from the policy
            self._state = np.concatenate([new_state, np.zeros_like(self.goal_pos), self.goal_pos])
        else:
            self._state = np.concatenate([new_state, self.goal_pos, self.goal_pos])

        self._iter += 1

        return self._state, reward, absorbing, {}

    def render(self, mode='human'):

        # goal area
        self._viewer.circle(self.goal_pos, self._goal_area_radius, color=(76, 141, 247))

        # goal
        if self.in_goal(self._state):
            self._viewer.circle(self.goal_pos, self._goal_radius, color=(124, 252, 0))
        elif self.in_goal_area(self._state):
            self._viewer.circle(self.goal_pos, self._goal_radius, color=(150, 150, 150))
        else:
            self._viewer.circle(self.goal_pos, self._goal_radius, color=(210, 43, 43))

        # point mass
        self._viewer.circle(self._state, 2.0, color=(0, 0, 0))

        self._viewer.display(self._dt/self._ff_render_factor)

    def get_viewer(self):
        return self._viewer

    def stop(self):
        self._viewer.close()

    def critic_state_mask(self):
        return np.concatenate([np.ones_like(self._state[:2], dtype=bool),
                               np.zeros_like(self.goal_pos, dtype=bool),
                               np.ones_like(self.goal_pos, dtype=bool)])

    def policy_state_mask(self):
        return np.concatenate([np.ones_like(self._state[:2], dtype=bool),
                               np.ones_like(self.goal_pos, dtype=bool),
                               np.zeros_like(self.goal_pos, dtype=bool)])

    def in_goal(self, state):
        return self.goal_pos[0] - self._goal_radius <= state[0] <= self.goal_pos[0] + self._goal_radius \
               and self.goal_pos[1] - self._goal_radius <= state[1] <= self.goal_pos[1] + self._goal_radius

    def in_goal_area(self, state):
        return self.goal_pos[0] - self._goal_area_radius <= state[0] <= self.goal_pos[0] + self._goal_area_radius \
               and self.goal_pos[1] - self._goal_area_radius <= state[1] <= self.goal_pos[1] + self._goal_area_radius

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

    mdp = PointMassPOMDP()
    for j in range(100):
        state = mdp.reset()
        print("State: ", state)
        for i in range(15):
            a = np.random.randn()
            state = mdp.step([a])
            print("State: ", state)
            mdp.render()
