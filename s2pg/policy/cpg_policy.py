from copy import deepcopy
import numpy as np
import torch.nn
from mushroom_rl.policy import Policy, ParametricPolicy, GaussianTorchPolicy
from mushroom_rl.policy.deterministic_policy import DeterministicPolicy

from itertools import chain


class CPGPolicy(Policy):

    def __init__(self, mdp_info, n_oscillators, dim_cpg_state, cpg_kernel, post_policy, sw=None, standardizer=None,
                 logging_iter=1000, init_r=0.0):

        assert isinstance(cpg_kernel, ParametricPolicy) or isinstance(cpg_kernel, GaussianTorchPolicy), \
            "CPG-Kernel needs to be a ParametricPolicy or GaussianTorchPolicy, but is %s." % type(cpg_kernel).__name__
        assert isinstance(post_policy, ParametricPolicy) or isinstance(post_policy, GaussianTorchPolicy),\
            "Post-Policy needs to be a ParametricPolicy or GaussianTorchPolicy, but is %s." % type(post_policy).__name__
        self.cpg_kernel = cpg_kernel
        self.post_policy = post_policy
        assert len(mdp_info.observation_space.shape) == 1, "N-D states currently not allowed."
        self._dim_env_state = mdp_info.observation_space.shape[0]
        self._dim_cpg_state = dim_cpg_state
        self._n_oscillators = n_oscillators
        self._init_r = init_r
        last_cpg_state = np.zeros(self._dim_cpg_state*self._n_oscillators, dtype=np.float32)
        last_cpg_state[self._r_mask] = self._init_r
        self._last_cpg_state = last_cpg_state
        self._psi_mask = np.arange(0, n_oscillators)
        self._r_mask = np.arange(n_oscillators, 2*n_oscillators)
        self._rd_mask = np.arange(2*n_oscillators, 3*n_oscillators)
        self.deterministic = isinstance(cpg_kernel, DeterministicPolicy) #TODO: Make more robust

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._stand = standardizer
        self._logging_iter = logging_iter
        self._iter = 0

        self._add_save_attr(
            cpg_kernel='mushroom',
            post_policy='mushroom',
            deterministic='primitive',
            _psi_mask='primitive',
            _r_mask='primitive',
            _rd_mask='primitive',
            _dim_env_state='primitive',
            _dim_cpg_state='primitive',
            _n_oscillators='primitive',
            _last_cpg_state='primitive',
            _init_r='primitive',
            _iter='primitive',
            _logging_iter='primitive'
        )

    def __call__(self, state, action, nstate):
        raise AttributeError("'CPGPolicy' object can not be called to calculate the pdf.")

    def draw_action(self, state):
        """ This function calculates the action based on the current cpg state (not the next one!)
            and the post policy. """

        if self._stand is not None:
            state = self._stand(state)

        self.cpg_kernel.deterministic = self.deterministic
        self.post_policy.deterministic = self.deterministic

        _, cpg_state = self.divide_state_to_env_cpg(state)
        action = self.post_policy.draw_action(cpg_state)
        self._last_cpg_state = self.cpg_kernel.draw_action(cpg_state)

        self._iter += 1

        return action

    def get_last_cpg_state(self):
        return deepcopy(self._last_cpg_state)

    def reset(self):
        last_cpg_state = np.zeros(self._dim_cpg_state*self._n_oscillators, dtype=np.float32)
        last_cpg_state[self._r_mask] = self._init_r
        self._last_cpg_state = last_cpg_state

    def divide_state_to_env_cpg(self, state):
        assert len(state.shape) == 1, "This function only divides single states and no batches."
        return state[0:self._dim_env_state], state[self._dim_env_state:]

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def diff_log(self, cpg_x, u, cpg_xn):
        d_log_cpg = self.cpg_kernel.diff_log(cpg_x, cpg_xn)
        d_log_pi = self.post_policy.diff_log(cpg_x, u)
        return np.concatenate([d_log_cpg, d_log_pi])

    def diff_log_batch(self, cpg_x, u, cpg_xn):
        d_log_cpg = self.cpg_kernel.diff_log_batch(cpg_x, cpg_xn)
        d_log_pi = self.post_policy.diff_log_batch(cpg_x, u)
        return np.concatenate([d_log_cpg, d_log_pi], axis=0)

    def get_weights(self):
        w_cpg = self.cpg_kernel.get_weights()
        w_pi = self.post_policy.get_weights()
        return np.concatenate([w_cpg, w_pi])

    def get_weight_names(self):
        names_cpg = self.cpg_kernel.get_weight_names()
        names_pi = self.post_policy.get_weight_names()
        return names_cpg + names_pi

    def set_weights(self, weights):
        self.cpg_kernel.set_weights(weights[0:self.cpg_kernel.weights_size])
        self.post_policy.set_weights(weights[self.cpg_kernel.weights_size:])

    @property
    def weights_size(self):
        return self.cpg_kernel.weights_size + self.post_policy.weights_size


class CPGPolicyStateDependent(CPGPolicy):

    def draw_action(self, state):
        """ This function calculates the action based on the current cpg state (not the next one!)
            and the post policy. """

        self.cpg_kernel.deterministic = self.deterministic
        self.post_policy.deterministic = self.deterministic

        _, cpg_state = self.divide_state_to_env_cpg(state)
        action = self.post_policy.draw_action(cpg_state)
        self._last_cpg_state = self.cpg_kernel.draw_action(state)

        self._iter += 1

        return action

    def diff_log(self, x, u, xn):
        cpg_xn = self.divide_state_to_env_cpg(xn)
        d_log_cpg = self.cpg_kernel.diff_log(x, cpg_xn)
        d_log_pi = self.post_policy.diff_log(x, u)
        return np.concatenate([d_log_cpg, d_log_pi])

    def diff_log_batch(self, x, u, xn):
        env_x, _ = self.divide_state_to_env_cpg_batch(x)
        _, cpg_xn = self.divide_state_to_env_cpg_batch(xn)
        # do some logging for debugging
        vbc = self.cpg_kernel._approximator.model.network.oscillator_ode._parameter_network(torch.tensor(env_x)).detach().cpu().numpy()
        v = vbc[:, 0]
        b = vbc[:, 1]
        c = vbc[:, 2]
        v_mean = np.mean(v)
        v_var = np.var(v)
        v_min = np.min(v)
        v_max = np.max(v)
        b_mean = np.mean(b)
        b_var = np.var(b)
        b_min = np.min(b)
        b_max = np.max(b)
        c_mean = np.mean(c)
        c_var = np.var(c)
        c_min = np.min(c)
        c_max = np.max(c)
        self._sw.add_scalar("v/mean", v_mean, self._iter)
        self._sw.add_scalar("v/var", v_var, self._iter)
        self._sw.add_scalar("v/min", v_min, self._iter)
        self._sw.add_scalar("v/max", v_max, self._iter)
        self._sw.add_scalar("b/mean", b_mean, self._iter)
        self._sw.add_scalar("b/var", b_var, self._iter)
        self._sw.add_scalar("b/min", b_min, self._iter)
        self._sw.add_scalar("b/max", b_max, self._iter)
        self._sw.add_scalar("c/mean", c_mean, self._iter)
        self._sw.add_scalar("c/var", c_var, self._iter)
        self._sw.add_scalar("c/min", c_min, self._iter)
        self._sw.add_scalar("c/max", c_max, self._iter)


        d_log_cpg = self.cpg_kernel.diff_log_batch(x, cpg_xn)
        d_log_pi = self.post_policy.diff_log_batch(x, u)
        return np.concatenate([d_log_cpg, d_log_pi], axis=0)


class CPGPolicyPPO(CPGPolicy):

    def __init__(self, use_cuda, cpg_use_env_state=True, **kwargs):
        super().__init__(**kwargs)
        self.use_cuda = use_cuda
        self._cpg_use_env_state = cpg_use_env_state

        self._add_save_attr(
            use_cuda='primitive')

    def diff_log(self, cpg_x, u, cpg_xn):
        raise AttributeError("This function is not supposed to be called for the CPGPolicyPPO class.")

    def diff_log_batch(self, cpg_x, u, cpg_xn):
        raise AttributeError("This function is not supposed to be called for the CPGPolicyPPO class.")

    def parameters(self):
        return chain(self.cpg_kernel.parameters(), self.post_policy.parameters())

    def entropy_t(self):
        # Todo: Create torch mixture distribution for both Gaussians and calculate entropy.
        raise AttributeError("This function is not supposed to be called for the CPGPolicyPPO class.")

    def log_prob_t(self, state, action):
        # Todo: Calculate the log prob of the Gaussian (currently not needed though).
        raise AttributeError("This function is not supposed to be called for the CPGPolicyPPO class.")

    def distribution_t(self):
        # Todo: Create torch mixture distribution for both Gaussians.
        raise AttributeError("Currently the distribution can not be calculated.")


class CPGPolicyTD3(ParametricPolicy):

    def __init__(self, mdp_info, mu_cpg, mu_post_pi, sigma_cpg, sigma_post_pi, n_oscillators, dim_cpg_state, low, high,
                 post_pi_state_dependent=False, post_pi_use_next_cpg_state=False, cpg_use_env_state=True,
                 init_r=0.0, use_stepwise_low_high=False):

        self._cpg_approximator = mu_cpg
        self._post_pi_approximator = mu_post_pi
        self._predict_params = dict()
        self._cpg_sigma = np.diag(np.ones(n_oscillators*dim_cpg_state) * sigma_cpg)
        self._post_pi_sigma = np.diag(np.ones(mdp_info.action_space.shape[0]) * sigma_post_pi)
        self.deterministic = False
        self._dim_env_state = mdp_info.observation_space.shape[0]
        self._dim_cpg_state = dim_cpg_state
        self._n_oscillators = n_oscillators
        self._psi_mask = np.arange(0, n_oscillators)
        self._r_mask = np.arange(n_oscillators, 2 * n_oscillators)
        self._rd_mask = np.arange(2 * n_oscillators, 3 * n_oscillators)
        self._low = low
        self._high = high
        self._low_cpg = self._cpg_approximator.model.network.low
        self._high_cpg = self._cpg_approximator.model.network.high
        self._post_pi_state_dependent = post_pi_state_dependent
        self._post_pi_use_next_cpg_state = post_pi_use_next_cpg_state
        self._cpg_use_env_state = cpg_use_env_state
        self._init_r = init_r
        last_cpg_state = np.zeros(self._dim_cpg_state*self._n_oscillators, dtype=np.float32)
        last_cpg_state[self._r_mask] = self._init_r
        self._last_cpg_state = last_cpg_state
        self._use_stepwise_low_high = use_stepwise_low_high

        self._add_save_attr(
            _cpg_approximator='mushroom',
            _post_pi_approximator='mushroom',
            _predict_params='pickle',
            _cpg_sigma='numpy',
            _post_pi_sigma='numpy',
            _low='numpy',
            _high='numpy',
            _dim_env_state='primitive',
            _dim_cpg_state='primitive',
            _n_oscillators='primitive',
            _last_cpg_state='numpy',
            _psi_mask='numpy',
            _r_mask='numpy',
            _rd_mask='numpy',
            deterministic='primitive',
            _post_pi_state_dependent="primitive",
            _post_pi_use_next_cpg_state="primitive",
            _cpg_use_env_state="primitive",
            _init_r='primitive',
            _use_stepwise_low_high='primitive'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def get_last_cpg_state(self):
        return deepcopy(self._last_cpg_state)

    def reset(self):
        norm_act_mean = self._cpg_approximator.model.network.norm_act_mean.detach().cpu().numpy()
        norm_act_delta = self._cpg_approximator.model.network.norm_act_delta.detach().cpu().numpy()
        last_cpg_state = np.zeros(self._dim_cpg_state*self._n_oscillators, dtype=np.float32)
        last_cpg_state[self._r_mask] = self._init_r
        # normalize state
        if self._cpg_approximator.model.network.use_normalized_cpg_state:
            last_cpg_state = (last_cpg_state - norm_act_mean) / norm_act_delta
        self._last_cpg_state = last_cpg_state

    def divide_state_to_env_cpg(self, state):
        assert len(state.shape) == 1, "This function only divides single states and no batches."
        return state[0:self._dim_env_state], state[self._dim_env_state:]

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def draw_action(self, state):
        env_state, cpg_state = self.divide_state_to_env_cpg(state)
        cpg_input = state if self._cpg_use_env_state else cpg_state
        next_cpg_state = np.reshape(self._cpg_approximator.predict(np.expand_dims(cpg_input, axis=0),
                                                                   **self._predict_params), -1)

        if not self._post_pi_use_next_cpg_state and not self._post_pi_state_dependent:
            action = np.reshape(self._post_pi_approximator.predict(np.expand_dims(cpg_state, axis=0),
                                                                   **self._predict_params), -1)
        elif self._post_pi_use_next_cpg_state and not self._post_pi_state_dependent:
            action = np.reshape(self._post_pi_approximator.predict(np.expand_dims(next_cpg_state, axis=0),
                                                                   **self._predict_params), -1)
        elif not self._post_pi_use_next_cpg_state and self._post_pi_state_dependent:
            action = np.reshape(self._post_pi_approximator.predict(np.expand_dims(state, axis=0),
                                                                   **self._predict_params), -1)
        else:
            post_pi_input = np.concatenate([env_state, next_cpg_state])
            action = np.reshape(self._post_pi_approximator.predict(np.expand_dims(post_pi_input, axis=0),
                                                                   **self._predict_params), -1)

        if self.deterministic:
            self._last_cpg_state = next_cpg_state
        else:
            if self._use_stepwise_low_high:
                low_cpg, high_cpg = self._cpg_approximator.model.network.get_state_dependent_lows_highs(cpg_state)
            else:
                low_cpg = self._low_cpg
                high_cpg = self._high_cpg
            noise_scaling = 0.5 * (high_cpg - low_cpg) if self._use_stepwise_low_high else 1.0
            cpg_state_eps = noise_scaling * np.random.normal(scale=np.sqrt(np.diag(self._cpg_sigma)), size=next_cpg_state.shape)
            last_cpg_state = np.clip(next_cpg_state + cpg_state_eps, np.squeeze(low_cpg), np.squeeze(high_cpg))
            self._last_cpg_state = np.squeeze(last_cpg_state)
            action = np.random.multivariate_normal(action, self._post_pi_sigma)

        # check if action is nan
        assert np.sum(np.isnan(action)) == 0, "Some of the predicted actions contain NANs."
        assert np.sum(np.isinf(action)) == 0, "Some of the predicted actions contain infs."

        return np.clip(action, self._low, self._high)

    def set_weights(self, weights):
        self._cpg_approximator.set_weights(weights[:self._cpg_approximator.weights_size])
        self._post_pi_approximator.set_weights(weights[self._cpg_approximator.weights_size:])

    def get_weights(self):
        cpg_weights = self._cpg_approximator.get_weights()
        post_pi_weights = self._post_pi_approximator.get_weights()
        return np.concatenate([cpg_weights, post_pi_weights])

    @property
    def weights_size(self):
        return self._cpg_approximator.weights_size + self._post_pi_approximator.weights_size


class StatefulPolicyPPO_joint(GaussianTorchPolicy):

    def __init__(self, action_dim, hidden_state_dim, init_hidden_state=None, **kwargs):

        super(StatefulPolicyPPO_joint, self).__init__(**kwargs)

        assert len(init_hidden_state) == hidden_state_dim, "init_hidden_state does not have" \
                                                           "the required dimensionality."
        self._action_dim_actual = action_dim    # the variable action_dim from the base class is rather an output dim
        self._hidden_state_dim = hidden_state_dim
        self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)
        self._init_hidden_state = init_hidden_state
        self.reset()

        self._add_save_attr(
            _action_dim_actual='primitive',
            _last_hidden_state='numpy',
            _init_hidden_state='numpy',
            _hidden_state_dim='primitive'
        )

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def reset(self):
        if self._init_hidden_state is not None:
            self._last_hidden_state = deepcopy(self._init_hidden_state)
        else:
            self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)

    def get_mean_and_covariance(self, state):
        mu_action, mu_next_hidden_state = self._mu(state, **self._predict_params, output_tensor=True)
        return torch.concat([mu_action, mu_next_hidden_state], dim=1), torch.diag(torch.exp(2 * self._log_sigma))

    def log_prob_t(self, state, action):
        raise NotImplementedError("This is forbidden here.")

    def log_prob_joint_t(self, state_hidden_state, action, next_hidden_state):
        action_next_hidden_state = torch.concat([action, next_hidden_state], dim=1)
        return self.distribution_t(state_hidden_state).log_prob(action_next_hidden_state)[:, None]

    def draw_action_t(self, state):
        state = torch.atleast_2d(state)
        if not self.deterministic:
            sample = torch.squeeze(self.distribution_t(state).sample().detach())
            action = sample[:self._action_dim_actual]
            next_hidden_state = sample[self._action_dim_actual:]
            self._last_hidden_state = next_hidden_state
            return action
        else:
            action, next_hidden_state = self._mu(state, **self._predict_params, output_tensor=True)
            action = torch.squeeze(action)
            next_hidden_state = torch.squeeze(next_hidden_state)
            self._last_hidden_state = next_hidden_state.detach().cpu().numpy()   # todo: this a _t function, so converting does not make a lot of sense. Maybe find a cleaner way.
            return action

    def divide_state_to_env_hidden_batch(self, states):
        return self._mu.model.network.divide_state_to_env_hidden_batch(states)
