import warnings
import numpy as np

from ..policy_gradient import StatefulPolicyGradient
from s2pg.policy import CPGPolicy


class GPOMDP_RS(StatefulPolicyGradient):
    """
    GPOMDP algorithm.
    "Infinite-Horizon Policy-Gradient Estimation". Baxter J. and Bartlett P. L..
    2001.
    """
    def __init__(self, mdp_info, policy, optimizer, features=None, sw=None):
        super().__init__(mdp_info, policy, optimizer, features)
        assert isinstance(policy, CPGPolicy)    # todo: is this still needed?
        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.list_sum_d_log_pi_ep = list()

        self.list_reward = list()
        self.list_reward_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        self.step_count = 0

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
            list_sum_d_log_pi_ep='pickle',
            list_reward='pickle',
            list_reward_ep='pickle',
            baseline_num='pickle',
            baseline_den='pickle',
            step_count='numpy'
        )

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

    def do_logging(self):
        # Todo: This is very extensive logging done temporarily only for debugging
        # get cpg params
        v = self.policy.cpg_kernel._approximator.model.network.oscillator_ode.v
        b = self.policy.cpg_kernel._approximator.model.network.oscillator_ode.b
        c = self.policy.cpg_kernel._approximator.model.network.oscillator_ode.c
        w = self.policy.cpg_kernel._approximator.model.network.oscillator_ode.w
        phi = self.policy.cpg_kernel._approximator.model.network.oscillator_ode.phi
        stds = np.exp(self.policy.cpg_kernel._logstd)
        self._sw.add_scalar("CPG Params/v1", v[0], self._iter)
        self._sw.add_scalar("CPG Params/b1", b[0], self._iter)
        self._sw.add_scalar("CPG Params/c1", c[0], self._iter)
        self._sw.add_scalar("CPG Params/w1", w[0, 0], self._iter)
        self._sw.add_scalar("CPG Params/phi1", phi[0, 0], self._iter)
        self._sw.add_scalar("CPG Params/v2", v[1], self._iter)
        self._sw.add_scalar("CPG Params/b2", b[1], self._iter)
        self._sw.add_scalar("CPG Params/c2", c[1], self._iter)
        self._sw.add_scalar("CPG Params/w2", w[0, 1], self._iter)
        self._sw.add_scalar("CPG Params/phi2", phi[0, 1], self._iter)
        for i, std in enumerate(stds):
            self._sw.add_scalar("CPG Params/std %d" % i, std, self._iter)

        # get policy params
        bias = self.policy.post_policy._approximator.model.network._bias
        stds = np.exp(self.policy.post_policy._logstd)
        self._sw.add_scalar("Post-Policy Params/bias0", bias[0], self._iter)
        self._sw.add_scalar("Post-Policy Params/bias1", bias[1], self._iter)
        for i, std in enumerate(stds):
            self._sw.add_scalar("Post-Policy Params/std %d" % i, std, self._iter)

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
        n_episodes = len(self.list_sum_d_log_pi_ep)
        grad_J_episode = list()
        for i in range(n_episodes):
            list_sum_d_log_pi = self.list_sum_d_log_pi_ep[i]
            list_reward = self.list_reward_ep[i]

            n_steps = len(list_sum_d_log_pi)

            gradient = np.zeros(self.policy.weights_size)

            for t in range(n_steps):
                step_grad = list_sum_d_log_pi[t]
                step_reward = list_reward[t]
                baseline = np.mean(self.baseline_num[t], axis=0) / np.mean(self.baseline_den[t], axis=0)
                baseline[np.logical_not(np.isfinite(baseline))] = 0.
                gradient += step_grad * (step_reward - baseline)

            grad_J_episode.append(gradient)

        gradJ = np.mean(grad_J_episode, axis=0)

        self.list_reward_ep = list()
        self.list_sum_d_log_pi_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        # do some logging
        # grad_J_cpg = gradJ[0:self.policy.cpg_kernel.weights_size]
        # grad_J_pi = gradJ[self.policy.cpg_kernel.weights_size:]
        #
        # param_names = self.policy.get_weight_names()
        # cpg_param_names = param_names[:self.policy.cpg_kernel.weights_size]
        # pi_param_names = param_names[self.policy.cpg_kernel.weights_size:]
        #
        # for i, p in zip(cpg_param_names, grad_J_cpg):
        #     self._sw.add_scalar("CPG Gradients/Grad Param %s" % i, p, self._iter)
        #
        # for i, p in zip(pi_param_names, grad_J_pi):
        #     self._sw.add_scalar("Pi Gradients/Grad Param %s" % i, p, self._iter)
        #
        # self.do_logging()

        self._iter += 1

        return gradJ

    def _step_update(self, x, u, r, xn):

        # get env and cpg state
        env_x, cpg_x = self.policy.divide_state_to_env_cpg(x)
        env_xn, cpg_xn = self.policy.divide_state_to_env_cpg(xn)

        discounted_reward = self.df * r
        self.list_reward.append(discounted_reward)

        d_log_pi = self.policy.diff_log(cpg_x, u, cpg_xn)
        self.sum_d_log_pi += d_log_pi

        self.list_sum_d_log_pi.append(self.sum_d_log_pi.copy())

        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)

        if self.step_count >= len(self.baseline_num):
            self.baseline_num.append(list())
            self.baseline_den.append(list())

        self.baseline_num[self.step_count].append(discounted_reward * squared_sum_d_log_pi)
        self.baseline_den[self.step_count].append(squared_sum_d_log_pi)

        self.step_count += 1

    def _episode_end_update(self):
        self.list_reward_ep.append(self.list_reward)
        self.list_reward = list()

        self.list_sum_d_log_pi_ep.append(self.list_sum_d_log_pi)
        self.list_sum_d_log_pi = list()

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)
        self.list_sum_d_log_pi = list()
        self.step_count = 0