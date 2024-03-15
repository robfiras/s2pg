import warnings
import numpy as np

from ..policy_gradient import StatefulPolicyGradient, BatchedStatefulPolicyGradient
from s2pg.policy import CPGPolicy
from mushroom_rl.utils.dataset import compute_J, parse_dataset


class REINFORCE_RS(StatefulPolicyGradient):
    """
    REINFORCE algorithm.
    "Simple Statistical Gradient-Following Algorithms for Connectionist
    Reinforcement Learning", Williams R. J.. 1992.
    """
    def __init__(self, mdp_info, policy, optimizer, features=None, sw=None):
        assert isinstance(policy, CPGPolicy)    # todo: is this still needed?
        super().__init__(mdp_info, policy, optimizer, features)
        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.baseline_num = list()
        self.baseline_den = list()

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

        # add the preprocessor to append the cpg state to the environment state
        self._preprocessors.append(self.append_cpg_state)

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._iter = 0

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        self._add_save_attr(
            sum_d_log_pi='numpy',
            list_sum_d_log_pi='pickle',
            baseline_num='pickle',
            baseline_den='pickle',
            _iter='primitive'
        )

    def add_preprocessor(self, preprocessor):
        # for now disable the preprocessor to ensure that appending the cpg state is always done at last
        raise AttributeError("This agent current does not support preprocessors.")

    def add_cpg_state_preprocessor(self):
        if len(self._preprocessors) == 0:   # only the cpg preprocessor is allowed for now, which is why we can check the length
            self._preprocessors.append(self.append_cpg_state)
        else:
            warnings.warn("CPG Preprocessor already included, and will be not added twice.")

    def append_cpg_state(self, x):
        # get latest cpg_state
        cpg_state = self.policy.get_last_cpg_state()
        return np.concatenate([x, cpg_state])

    def _compute_gradient(self, J):
        baseline = np.mean(self.baseline_num, axis=0) / np.mean(self.baseline_den, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.
        grad_J_episode = list()
        for i, J_episode in enumerate(J):
            sum_d_log_pi = self.list_sum_d_log_pi[i]
            grad_J_episode.append(sum_d_log_pi * (J_episode - baseline))

        grad_J = np.mean(grad_J_episode, axis=0)
        self.list_sum_d_log_pi = list()
        self.baseline_den = list()
        self.baseline_num = list()

        # do some logging
        grad_J_cpg = grad_J[0:self.policy.cpg_kernel.weights_size]
        grad_J_pi = grad_J[self.policy.cpg_kernel.weights_size:]

        param_names = self.policy.get_weight_names()
        cpg_param_names = param_names[:self.policy.cpg_kernel.weights_size]
        pi_param_names = param_names[self.policy.cpg_kernel.weights_size:]

        for i, p in zip(cpg_param_names, grad_J_cpg):
            self._sw.add_scalar("CPG Gradients/Grad Param %s" % i, p, self._iter)

        for i, p in zip(pi_param_names, grad_J_pi):
            self._sw.add_scalar("Pi Gradients/Grad Param %s" % i, p, self._iter)

        # do rest of logging
        #self.do_logging()

        self._iter += 1

        return grad_J

    def _step_update(self, x, u, r, xn):
        # get env and cpg state
        env_x, cpg_x = self.policy.divide_state_to_env_cpg(x)
        env_xn, cpg_xn = self.policy.divide_state_to_env_cpg(xn)

        # calculate the step-based gradients
        d_log_pi = self.policy.diff_log(cpg_x, u, cpg_xn)
        self.sum_d_log_pi += d_log_pi

    def _episode_end_update(self):
        self.list_sum_d_log_pi.append(self.sum_d_log_pi)
        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)
        self.baseline_num.append(squared_sum_d_log_pi * self.J_episode)
        self.baseline_den.append(squared_sum_d_log_pi)

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)


class BatchedREINFORCE_RS(BatchedStatefulPolicyGradient):
    """
    REINFORCE algorithm.
    "Simple Statistical Gradient-Following Algorithms for Connectionist
    Reinforcement Learning", Williams R. J.. 1992.
    """
    def __init__(self, mdp_info, policy, optimizer, features=None, sw=None, use_env_state=True, logger=None, int_log=1):
        super().__init__(mdp_info, policy, optimizer, features)
        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.baseline_num = list()
        self.baseline_den = list()
        self._use_env_state = use_env_state
        self._logger = logger
        self._int_log = int_log

        # add the preprocessor to append the cpg state to the environment state
        self._preprocessors.append(self.append_hidden_state)

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._iter = 0

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        self._add_save_attr(
            sum_d_log_pi='numpy',
            list_sum_d_log_pi='pickle',
            baseline_num='pickle',
            baseline_den='pickle',
            _iter='primitive'
        )

    def add_preprocessor(self, preprocessor):
        # for now disable the preprocessor to ensure that appending the hidden state is always done at last
        raise AttributeError("This agent current does not support preprocessors.")

    def add_hidden_state_preprocessor(self):
        if len(self._preprocessors) == 0:   # only the hidden preprocessor is allowed for now, which is why we can check the length
            self._preprocessors.append(self.append_hidden_state)
        else:
            warnings.warn("Hidden Preprocessor already included, and will be not added twice.")

    def append_hidden_state(self, x):
        # get latest cpg_state
        hidden_state = self.policy.get_last_hidden_state()
        return np.concatenate([x, hidden_state])

    def _compute_gradient(self, J):
        baseline = np.mean(self.baseline_num, axis=0) / np.mean(self.baseline_den, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.
        grad_J_episode = list()
        for i, J_episode in enumerate(J):
            sum_d_log_pi = self.list_sum_d_log_pi[i]
            grad_J_episode.append(sum_d_log_pi * (J_episode - baseline))

        grad_J = np.mean(grad_J_episode, axis=0)
        self.list_sum_d_log_pi = list()
        self.baseline_den = list()
        self.baseline_num = list()

        # do some logging
        entropy = self.policy.entropy()

        if self._sw and self._iter % self._int_log == 0:
            var = np.mean(np.var(grad_J_episode, axis=0))
            self._sw.add_scalar("Policy/Mean Gradient", np.mean(grad_J_episode), self._iter)
            self._sw.add_scalar("Policy/Var Gradient", var, self._iter)
            if self._logger is not None:
                for logger in self._logger:
                    logger.log_numpy(Var_Gradient=var)
            self._sw.add_scalar("Entropy/Entropy", entropy, self._iter)
            self._sw.add_scalar("Entropy/Mean std ", np.mean(np.exp(self.policy._logstd)),
                                self._iter)
        self._iter += 1

        return grad_J

    def _all_step_updates(self, dataset):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)

        prev_u = self.get_prev_action(u, last, absorbing)

        if not self._use_env_state:
            # get env and cpg states
            _, x = self.policy.divide_state_to_env_hidden_batch(x)
            _, xn = self.policy.divide_state_to_env_hidden_batch(xn)

        # calculate the sum for each episode
        split_idx = np.squeeze(np.argwhere(last == 1))[:-1]+1

        # divide into episodes
        x_episodes = np.split(x, split_idx)
        xn_episodes = np.split(xn, split_idx)
        u_episodes = np.split(u, split_idx)
        prev_u_episodes = np.split(prev_u, split_idx)

        # calculate the J's
        Js = compute_J(dataset, gamma=self.mdp_info.gamma)

        for x_e, xn_e, u_e, prev_u_e, J in zip(x_episodes, xn_episodes, u_episodes, prev_u_episodes, Js):
            # calculate the step-based gradients
            sum_d_log_pi = self.policy.diff_log_batch(x_e, u_e, xn_e, prev_u_e)
            self.list_sum_d_log_pi.append(sum_d_log_pi)
            squared_sum_d_log_pi = np.square(sum_d_log_pi)
            self.baseline_num.append(squared_sum_d_log_pi * J)
            self.baseline_den.append(squared_sum_d_log_pi)

        return Js

    @staticmethod
    def get_prev_action(actions, lasts, absorbings):
        pa = list()
        for i in range(len(actions)):
            if i == 0 or lasts[i - 1] or absorbings[i - 1]:
                pa.append(np.zeros(len(actions[0])))
            else:
                pa.append(actions[i - 1])

        return np.array(pa)
