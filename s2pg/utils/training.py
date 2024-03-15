import os
import numpy as np
from copy import deepcopy

from mushroom_rl.utils.dataset import compute_J, compute_episodes_length


class BestAgentSaver:

    def __init__(self, save_path, n_epochs_save=10, save_replay_memory=False):
        self.best_curr_agent = None
        self.save_path = save_path
        self.n_epochs_save = n_epochs_save
        self.last_save = 0
        self.epoch_counter = 0
        self.best_J_since_last_save = -float('inf')
        self.save_replay_memory = save_replay_memory

    def save(self, agent, J):

        if self.n_epochs_save != -1:
            if J > self.best_J_since_last_save:
                self.best_J_since_last_save = J
                # if the agent has a replay memory that should not be saved, we can save memory by not copying it,
                # i.e., temporarily removing it from the current agent and then giving it back.
                # Todo: add a mode to choose whether to do a full save or not. That should decide whether to
                #   keep the buffer or not.
                mem = None
                tmp_store_mem = hasattr(agent, '_replay_memory') and not self.save_replay_memory
                if tmp_store_mem:
                    mem = agent._replay_memory
                    agent._replay_memory = None
                self.best_curr_agent = (deepcopy(agent), J, self.epoch_counter)
                if tmp_store_mem:
                    agent._replay_memory = mem

            if self.last_save + self.n_epochs_save <= self.epoch_counter:
                self.save_curr_best_agent()

            self.epoch_counter += 1

    def save_curr_best_agent(self):

        if self.best_curr_agent is not None:
            path = os.path.join(self.save_path, 'agent_epoch_%d_J_%f.msh' % (self.best_curr_agent[2],
                                                                             self.best_curr_agent[1]))
            self.best_curr_agent[0].save(path, full_save=True)
            self.best_curr_agent = None
            self.best_J_since_last_save = -float('inf')
            self.last_save = self.epoch_counter

    def save_agent(self,  agent, J):
        path = os.path.join(self.save_path, 'agent_J_%f.msh' % J)
        agent.save(path, full_save=True)


def do_evaluation(mdp, core, n_epochs_eval, sw, i, min_R_J=-1000.0, return_dataset=False, render=False):

    test_dataset = core.evaluate(n_episodes=n_epochs_eval, render=render)
    J = np.mean(compute_J(test_dataset, gamma=mdp.info.gamma))
    R = np.mean(compute_J(test_dataset))
    L = np.mean(compute_episodes_length(test_dataset))
    # clip to minimum logging reward
    J = np.maximum(J, min_R_J)
    R = np.maximum(R, min_R_J)
    sw.add_scalar("Evaluation/Cumulated undiscounted Reward (stoch.)", R, i)
    sw.add_scalar("Evaluation/Cumulated discounted Reward (J) (stoch.)", J, i)
    sw.add_scalar("Evaluation/Episode Length (stoch.)", L, i)
    if return_dataset:
        return J, R, L, test_dataset
    else:
        return J, R, L


def do_evaluation_deterministic(mdp, core, n_epochs_eval, sw, i, min_R_J=-1000.0, return_dataset=False, render=False):
    core.agent.policy.deterministic = True
    test_dataset = core.evaluate(n_episodes=n_epochs_eval, render=render)
    J = np.mean(compute_J(test_dataset, gamma=mdp.info.gamma))
    R = np.mean(compute_J(test_dataset))
    L = np.mean(compute_episodes_length(test_dataset))
    # clip to minimum logging reward
    J = np.maximum(J, min_R_J)
    R = np.maximum(R, min_R_J)
    sw.add_scalar("Evaluation/Cumulated undiscounted Reward (deter.)", R, i)
    sw.add_scalar("Evaluation/Cumulated discounted Reward (J) (deter.)", J, i)
    sw.add_scalar("Evaluation/Episode Length (deter.)", L, i)
    core.agent.policy.deterministic = False

    if return_dataset:
        return J, R, L, test_dataset
    else:
        return J, R, L
