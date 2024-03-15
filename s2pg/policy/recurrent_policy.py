import numpy as np
import torch
from copy import deepcopy
from mushroom_rl.policy import Policy, ParametricPolicy, GaussianTorchPolicy
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from s2pg.policy import DiagonalGaussianPolicy


class RecurrentPolicy(ParametricPolicy):

    def __init__(self, mdp_info, mu_approximator, std_action, std_hidden_state, dim_hidden_state, use_cuda=False):

        self._mu_approximator = mu_approximator
        self._std_action = std_action
        self._std_hidden_state = std_hidden_state
        self._predict_params = dict()       # todo: for now deactivated
        self.deterministic = False
        self._dim_env_state = mdp_info.observation_space.shape[0]
        self._dim_hidden_state = dim_hidden_state
        self._last_hidden_state = np.zeros(self._dim_hidden_state, dtype=np.float32)
        self._last_action = np.zeros(mdp_info.action_space.shape[0], dtype=np.float32)
        self._mdp_info = mdp_info
        self.use_cuda = use_cuda

        self._add_save_attr(
            _mu_approximator='mushroom',
            _std_action='primitive',
            _std_hidden_state='primitive',
            _predict_params='pickle',
            _dim_env_state='primitive',
            _dim_hidden_state='primitive',
            _last_action='numpy',
            _last_hidden_state='numpy',
            _mdp_info='pickle',
            deterministic='primitive',
            use_cuda='primitive'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def reset(self):
        self._last_hidden_state = np.zeros(self._dim_hidden_state, dtype=np.float32)
        self._last_action = np.zeros(self._mdp_info.action_space.shape[0], dtype=np.float32)

    def draw_action(self, state):
        raise NotImplementedError

    def set_weights(self, weights):
        self._mu_approximator.set_weights(weights)

    def get_weights(self):
        return self._mu_approximator.get_weights()

    @property
    def weights_size(self):
        return self._mu_approximator.weight_size

    def divide_state_to_env_hidden_batch(self, states):
        return self._mu_approximator.model.network.divide_state_to_env_hidden_batch(states)


class RecurrentPolicyTD3(RecurrentPolicy):

    def __init__(self, low, high, **kwargs):
        super(RecurrentPolicyTD3, self).__init__(**kwargs)

        self._low = low
        self._high = high
        self._low_hidden, self._high_hidden = self.get_low_high_hidden_state()

        self._add_save_attr(
            _low='numpy',
            _high='numpy',
            _low_hidden='numpy',
            _high_hidden='numpy'
        )

    def get_low_high_hidden_state(self):
        return self._mu_approximator.model.network.get_low_high_hidden_state()

    def draw_action(self, state):

        prev_action = torch.tensor(self.get_last_action())
        action, next_hidden_state = self._mu_approximator.model.network(to_float_tensor(state),
                                                                        prev_action)
        action = action.detach().cpu().numpy()
        next_hidden_state = np.squeeze(next_hidden_state.detach().cpu().numpy())

        if self.deterministic:
            self._last_hidden_state = next_hidden_state
            self._last_action = deepcopy(action)
        else:
            hidden_state_eps = np.random.normal(scale=self._std_hidden_state, size=next_hidden_state.shape)
            self._last_hidden_state = np.clip(next_hidden_state + hidden_state_eps, self._low_hidden, self._high_hidden)
            action_eps = np.random.normal(scale=self._std_action, size=action.shape)
            action += action_eps
            self._last_action = deepcopy(action)

        # check if action is nan
        assert np.sum(np.isnan(action)) == 0, "Some of the predicted actions contain NANs."
        assert np.sum(np.isinf(action)) == 0, "Some of the predicted actions contain infs."

        return np.clip(action, self._low, self._high)

    @property
    def low_hidden(self):
        return self._low_hidden

    @property
    def high_hidden(self):
        return self._high_hidden


class RecurrentPolicyTD3_preprocess(RecurrentPolicyTD3):

    def __init__(self, state_mask, **kwargs):

        super().__init__(**kwargs)
        self._state_mask = state_mask

        self._add_save_attr(
            _state_mask='primitive'
        )

    def draw_action(self, state):
        state = state[self._state_mask]
        return super().draw_action(state)


class TD3WindowPolicy(RecurrentPolicyTD3_preprocess):

    def __init__(self, window_length, **kwargs):
        self._window_length = window_length
        super(TD3WindowPolicy, self).__init__(std_hidden_state=0.0, **kwargs)
        self._action_dim = len(self._low)

    def draw_action(self, state):

        state = np.atleast_2d(state[self._state_mask])
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        self.add_to_window(state, prev_a)
        current_window_obs = np.expand_dims(self._window_obs, axis=0)
        current_window_act = np.expand_dims(self._window_act, axis=0)

        prev_action = torch.tensor(self.get_last_action())
        action, _ = self._mu_approximator.model.network(to_float_tensor(current_window_obs),
                                                        to_float_tensor(current_window_act))

        next_hidden_state = torch.zeros(self._dim_hidden_state)

        action = action.detach().cpu().numpy()

        if self.deterministic:
            self._last_hidden_state = next_hidden_state
            self._last_action = deepcopy(action)
        else:
            self._last_hidden_state = next_hidden_state
            action_eps = np.random.normal(scale=self._std_action, size=action.shape)
            action += action_eps
            self._last_action = deepcopy(action)

        # check if action is nan
        assert np.sum(np.isnan(action)) == 0, "Some of the predicted actions contain NANs."
        assert np.sum(np.isinf(action)) == 0, "Some of the predicted actions contain infs."

        return np.clip(action, self._low, self._high)

    def reset(self):
        super(TD3WindowPolicy, self).reset()

        self._window_obs = np.zeros((self._window_length, np.sum(self._state_mask)))
        self._window_act = np.zeros((self._window_length, self._action_dim))
        self._curr_wind_ind = 0

    def add_to_window(self, state, prev_action):
        if self._curr_wind_ind == self._window_length:
            self._window_obs = np.delete(self._window_obs, 0, 0)
            self._window_obs = np.concatenate([self._window_obs, state])
            self._window_act = np.delete(self._window_act, 0, 0)
            self._window_act = np.concatenate([self._window_act, prev_action])
        else:
            self._window_obs[self._curr_wind_ind] = state[0]
            self._window_act[self._curr_wind_ind] = prev_action[0]
            self._curr_wind_ind += 1

    def get_low_high_hidden_state(self):
        return None, None


class RecurrentPolicyTD3BPTT(RecurrentPolicy):

    def __init__(self, low, high, **kwargs):
        if "std_hidden_state" in kwargs.keys():
            assert kwargs["std_hidden_state"] == 0.0
        else:
            kwargs["std_hidden_state"] = 0.0

        super(RecurrentPolicyTD3BPTT, self).__init__(**kwargs)

        self._low = low
        self._high = high

        self._add_save_attr(
            _low='numpy',
            _high='numpy',
            _low_hidden='numpy',
            _high_hidden='numpy'
        )

    def draw_action(self, state):
        state = np.atleast_2d(state)
        lengths = torch.tensor([1])
        prev_actions = torch.tensor(self.get_last_action()).view(1, 1, -1)
        action, next_hidden_state = self._mu_approximator.model.network(to_float_tensor(state).view(1, 1, -1),
                                                                        to_float_tensor(prev_actions),
                                                                        lengths)
        action = np.squeeze(action.detach().cpu().numpy())
        self._last_hidden_state = np.squeeze(next_hidden_state.detach().cpu().numpy())

        if not self.deterministic:
            action_eps = np.random.normal(scale=self._std_action, size=action.shape)
            action += action_eps
            self._last_action = deepcopy(action)
        else:
            self._last_action = deepcopy(action)

        # check if action is nan
        assert np.sum(np.isnan(action)) == 0, "Some of the predicted actions contain NANs."
        assert np.sum(np.isinf(action)) == 0, "Some of the predicted actions contain infs."

        return np.clip(action, self._low, self._high)


class RecurrentPolicyTD3BPTT_preprocess(RecurrentPolicyTD3BPTT):

    def __init__(self, state_mask, **kwargs):

        super().__init__(**kwargs)
        self._state_mask = state_mask

        self._add_save_attr(
            _state_mask='primitive'
        )

    def draw_action(self, state):
        state = state[self._state_mask]
        return super().draw_action(state)


class RecurrentPolicyPPOBPTT(GaussianTorchPolicy):

    def __init__(self, mdp_info, dim_hidden_state, log_std_min=-20, log_std_max=2, **kwargs):

        super(RecurrentPolicyPPOBPTT, self).__init__( **kwargs)

        self._dim_env_state = mdp_info.observation_space.shape[0]
        self._dim_hidden_state = dim_hidden_state
        self._last_hidden_state = np.zeros(self._dim_hidden_state, dtype=np.float32)
        self._last_action = np.zeros(mdp_info.action_space.shape[0], dtype=np.float32)
        self._mdp_info = mdp_info
        self.deterministic = False

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_action='primitive',
            _sigma_hidden_state='primitive',
            _predict_params='pickle',
            _dim_env_state='primitive',
            _dim_hidden_state='primitive',
            _last_action='numpy',
            _last_hidden_state='numpy',
            mdp_info='pickle',
            deterministic='primitive',
            use_cuda='primitive'
        )

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def reset(self):
        self._last_hidden_state = np.zeros(self._dim_hidden_state, dtype=np.float32)
        self._last_action = np.zeros(self._mdp_info.action_space.shape[0], dtype=np.float32)

    def draw_action(self, state):
        with torch.no_grad():
            a = self.draw_action_t(state)
        return torch.squeeze(a, dim=0).detach().cpu().numpy()

    def draw_action_t(self, state):
        prev_action = torch.tensor(self.get_last_action()).view(1, 1, -1)
        state = to_float_tensor(np.atleast_2d(state), self._use_cuda).view(1, 1, -1)
        lengths = torch.tensor([1])
        if not self.deterministic:
            dist, nhidden = self.distribution_nhidden_t(state, prev_action, lengths)
            self._last_hidden_state = np.squeeze(nhidden.detach().cpu().numpy())
            action = dist.sample().detach()
            self._last_action = np.squeeze(action.cpu().numpy())
            return action
        else:
            mu, nhidden = self._mu(state, prev_action, lengths, **self._predict_params, output_tensor=True)
            self._last_hidden_state = np.squeeze(nhidden.detach().cpu().numpy())
            action = mu.detach()
            self._last_action = np.squeeze(action.cpu().numpy())
            return action

    def log_prob_t(self, state, action, prev_action, lengths):
        return self.distribution_t(state, prev_action, lengths).log_prob(action.squeeze())[:, None]

    def entropy_t(self, state=None):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma)

    def distribution(self, state, prev_action, lengths):
        s = to_float_tensor(state, self._use_cuda)
        prev_a = to_float_tensor(prev_action, self._use_cuda)

        return self.distribution_t(s, prev_a, lengths)

    def distribution_t(self, state, prev_action, lengths):
        mu, sigma, _ = self.get_mean_and_covariance_and_hidden_state(state, prev_action, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def distribution_nhidden_t(self, state, prev_action, lengths):
        mu, sigma, nhidden_state = self.get_mean_and_covariance_and_hidden_state(state, prev_action, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma), nhidden_state

    def get_mean_and_covariance_and_hidden_state(self, state, prev_action, lengths):
        mu, next_hidden_state = self._mu(state, prev_action, lengths, **self._predict_params, output_tensor=True)

        # Bound the log_std
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min(), self._log_std_max())

        covariance = torch.diag(torch.exp(2 * log_sigma))
        return mu, covariance, next_hidden_state


class RecurrentPolicyPPOBPTT_preprocess(RecurrentPolicyPPOBPTT):

    def __init__(self, state_mask, **kwargs):

        super().__init__(**kwargs)
        self._state_mask = state_mask

        self._add_save_attr(
            _state_mask='primitive'
        )

    def draw_action(self, state):
        state = state[self._state_mask]
        return super().draw_action(state)


class PPOWindowPolicy(RecurrentPolicyPPOBPTT):

    def __init__(self, window_length, state_mask=None, **kwargs):

        self._state_mask = state_mask
        self._window_length = window_length

        super(PPOWindowPolicy, self).__init__(**kwargs)

    def draw_action(self, state):
        if self._state_mask is not None:
            state = state[self._state_mask]
        with torch.no_grad():
            a = self.draw_action_t(state)
        return torch.squeeze(a, dim=0).detach().cpu().numpy()

    def draw_action_t(self, state):
        prev_action = np.atleast_2d(self.get_last_action())
        state = np.atleast_2d(state)
        self.add_to_window(state, prev_action)

        current_window_obs = to_float_tensor(np.expand_dims(self._window_obs, axis=0))
        current_window_act = to_float_tensor(np.expand_dims(self._window_act, axis=0))
        lengths = torch.tensor([1])

        if not self.deterministic:
            dist = self.distribution_t(current_window_obs, current_window_act, lengths)
            action = dist.sample().detach()
            self._last_action = np.squeeze(action.cpu().numpy())
            return action
        else:
            mu = self._mu(current_window_obs, current_window_act, lengths,
                                   **self._predict_params, output_tensor=True)
            action = mu.detach()
            self._last_action = np.squeeze(action.cpu().numpy())
            return action

    def distribution_t(self, state, prev_action, lengths):
        mu, sigma = self.get_mean_and_covariance(state, prev_action, lengths)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def get_mean_and_covariance(self, state, prev_action, lengths):
        mu = self._mu(state, prev_action, lengths, **self._predict_params, output_tensor=True)

        # Bound the log_std
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min(), self._log_std_max())

        covariance = torch.diag(torch.exp(2 * log_sigma))
        return mu, covariance

    def reset(self):
        super(PPOWindowPolicy, self).reset()

        self._window_obs = np.zeros((self._window_length, np.sum(self._state_mask)))
        self._window_act = np.zeros((self._window_length, self._action_dim))
        self._curr_wind_ind = 0

    def add_to_window(self, state, prev_action):
        if self._curr_wind_ind == self._window_length:
            self._window_obs = np.delete(self._window_obs, 0, 0)
            self._window_obs = np.concatenate([self._window_obs, state])
            self._window_act = np.delete(self._window_act, 0, 0)
            self._window_act = np.concatenate([self._window_act, prev_action])
        else:
            self._window_obs[self._curr_wind_ind] = state[0]
            self._window_act[self._curr_wind_ind] = prev_action[0]
            self._curr_wind_ind += 1


class StatefulPolicyPPO_joint_prev_a(GaussianTorchPolicy):

    def __init__(self, action_dim, hidden_state_dim, log_std_min=-20, log_std_max=2, init_hidden_state=None, **kwargs):

        super(StatefulPolicyPPO_joint_prev_a, self).__init__(action_dim=action_dim, **kwargs)

        if init_hidden_state:
            assert len(init_hidden_state) == hidden_state_dim, "init_hidden_state does not have" \
                                                               "the required dimensionality."
        self._action_dim_actual = action_dim    # the variable action_dim from the base class is rather an output dim
        self._hidden_state_dim = hidden_state_dim
        self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)
        self._last_action = np.zeros(self._action_dim_actual, dtype=np.float32)
        self._init_hidden_state = init_hidden_state
        self._last_cpg_state = None
        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)
        self.reset()
        self.deterministic = False

        self._add_save_attr(
            _action_dim_actual='primitive',
            _last_hidden_state='numpy',
            deteriministic='primitive'
        )

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def reset(self):
        if self._init_hidden_state is not None:
            self._last_hidden_state = deepcopy(self._init_hidden_state)
        else:
            self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)

    def get_mean_and_covariance(self, state, prev_action):
        lengths = torch.ones(state.shape[0], dtype=torch.long) # we use the same networks as for bptt_pI, but they require sequences, so we pass sequences with lengths 1
        mu_action, mu_next_hidden_state = self._mu(state, prev_action, lengths, **self._predict_params, output_tensor=True)
        mu_next_hidden_state = torch.squeeze(mu_next_hidden_state, dim=1)

        # Bound log std
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min(), self._log_std_max())

        return torch.concat([mu_action, mu_next_hidden_state], dim=1), torch.diag(torch.exp(2 * log_sigma))

    def distribution(self, state, prev_action):
        s = to_float_tensor(state, self._use_cuda)
        prev_action = to_float_tensor(prev_action)
        return self.distribution_t(s, prev_action)

    def distribution_t(self, state, prev_action):
        mu, sigma = self.get_mean_and_covariance(state, prev_action)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def log_prob_t(self, state, action):
        raise NotImplementedError("This is forbidden here.")

    def log_prob_joint_t(self, state_hidden_state, prev_action, action, next_hidden_state):
        action_next_hidden_state = torch.concat([action, next_hidden_state], dim=1)
        return self.distribution_t(state_hidden_state, prev_action).log_prob(action_next_hidden_state)[:, None]

    def draw_action_t(self, state):
        state = state.view(1, 1, -1)
        prev_action = torch.tensor(self.get_last_action()).view(1, 1, -1)
        if not self.deterministic:
            sample = torch.squeeze(self.distribution_t(state, prev_action).sample().detach())
            action = sample[:self._action_dim_actual]
            next_hidden_state = sample[self._action_dim_actual:]
            self._last_hidden_state = next_hidden_state
            self._last_action = action.detach().cpu().numpy()
            return action
        else:
            lengths = torch.ones(state.shape[0],
                                 dtype=torch.long)  # we use the same networks as for bptt_pI, but they require sequences, so we pass sequences with lengths 1
            action, next_hidden_state = self._mu(state, prev_action, lengths, **self._predict_params, output_tensor=True)
            action = torch.squeeze(action)
            next_hidden_state = torch.squeeze(next_hidden_state)
            self._last_hidden_state = next_hidden_state.detach().cpu().numpy()   # todo: this a _t function, so converting does not make a lot of sense. Maybe find a cleaner way.
            self._last_action = action.detach().cpu().numpy()
            return action

    def divide_state_to_env_hidden_batch(self, states):
        return self._mu.model.network.divide_state_to_env_hidden_batch(states)


class StatefulPolicyPPO_joint_prev_a_preprocess(StatefulPolicyPPO_joint_prev_a):

    def __init__(self, state_mask, **kwargs):

        super().__init__(**kwargs)
        self._state_mask = state_mask

        self._add_save_attr(
            _state_mask='primitive'
        )

    def draw_action(self, state):
        state = state[self._state_mask]
        return super().draw_action(state)


class SACRecurrentPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic
    algorithm. The policy is a Gaussian policy squashed by a tanh.
    This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for
    the internals calculations of the SAC algorithm.

    """
    def __init__(self, approximator, hidden_state_dim, action_dim, lows, highs,
                 log_std_min, log_std_max, state_mask=None, init_hidden_state=None):

        self.use_mean = False  # if true the mean action is taken instead of sampling from Gaussian
        self._approximator = approximator
        self._output_shape = self._approximator.model.network.output_shape
        self._half_out_shape = self._output_shape // 2
        assert type(self._output_shape) == int, "Output shape needs to be an integer."
        assert 2 * self._half_out_shape == self._output_shape, "Output shape needs to be an even number."

        self._delta_a = to_float_tensor(.5 * (highs - lows), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (highs + lows), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._state_mask = state_mask

        self._eps_log_prob = 1e-6

        use_cuda = self._approximator.model.use_cuda

        self._hidden_state_dim = hidden_state_dim
        self._action_dim = action_dim
        self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)
        self._last_action = np.zeros(self._action_dim, dtype=np.float32)
        self._init_hidden_state = init_hidden_state
        self.reset()

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive',
            _output_shape='primitive',
            _half_out_shape='primitive',
            _hidden_state_dim='numpy',
            _action_dim='numpy',
            _last_hidden_state='numpy',
            _last_action='numpy',
            _init_hidden_state='numpy',
            _state_mask='primitive'
        )

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def reset(self):
        if self._init_hidden_state is not None:
            self._last_hidden_state = deepcopy(self._init_hidden_state)
        else:
            self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        if self._state_mask is not None:
            state = state[self._state_mask]
        state = np.atleast_2d(state)
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        out = self.compute_action_hidden_and_log_prob_t(state, prev_a, compute_log_prob=False).detach().cpu().numpy()
        action, next_hidden_state = self.divide_action_hidden_state(out)
        action = np.atleast_1d(np.squeeze(action))
        next_hidden_state = np.atleast_1d(np.squeeze(next_hidden_state))
        self._last_action = deepcopy(action)
        self._last_hidden_state = deepcopy(next_hidden_state)
        return action

    def compute_action_hidden_and_log_prob(self, state, prev_a, lengths=None):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        out, log_prob = self.compute_action_hidden_and_log_prob_t(state, prev_a, lengths)
        return out.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_hidden_and_log_prob_t(self, state, prev_a, compute_log_prob=True, return_dist=False,
                                             lengths=None):

        batch_size = state.shape[0]
        if len(state.shape) == 3:
            # we are using bptt_pI, so we need sequences
            assert lengths is not None
            assert len(prev_a.shape) == 3
        else:
            # we are not using bptt_pI, but we extend the dims to use the same bptt_pI network with sequence lengths 1
            state = state.reshape(batch_size, 1, -1)
            prev_a = prev_a.reshape(batch_size, 1, -1)
            lengths = np.ones(batch_size, dtype=int)

        dist = self.distribution(state.astype(np.float32), prev_a.astype(np.float32), lengths)
        out_raw = dist.rsample()
        out = torch.tanh(out_raw)
        out_true = out * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(out_raw).sum(dim=1)
            log_prob -= torch.log(1. - out.pow(2) + self._eps_log_prob).sum(dim=1)
            if return_dist:
                return out_true, log_prob, dist
            else:
                return out_true, log_prob
        else:
            if return_dist:
                return out_true, dist
            else:
                return out_true

    def compute_log_prob_separate(self, state, prev_a, lengths=None):
        out_true, log_prob_action, log_prob_next_hidden_state = self.compute_log_prob_separate_t(state, prev_a, lengths)
        out_true = out_true.detach().cpu().numpy()
        log_prob_action = log_prob_action.detach().cpu().numpy()
        log_prob_next_hidden_state = log_prob_next_hidden_state.detach().cpu().numpy()
        return out_true, log_prob_action, log_prob_next_hidden_state

    def compute_log_prob_separate_t(self, state, prev_a, lengths=None):

        if lengths is None:
            batch_size = state.shape[0]
            state = state.reshape(batch_size, 1, -1)
            prev_a = prev_a.reshape(batch_size, 1, -1)
            lengths = np.ones(batch_size, dtype=int)

        if type(state) == torch.Tensor:
            state = state.float()
        else:
            state = state.astype(np.float32)

        if type(prev_a) == torch.Tensor:
            prev_a = prev_a.float()
        else:
            prev_a = prev_a.astype(np.float32)

        dist = self.distribution(state, prev_a, lengths)
        out_raw = dist.rsample()
        out = torch.tanh(out_raw)
        out_true = out * self._delta_a + self._central_a
        log_prob = dist.log_prob(out_raw)

        action, next_hidden_state = self.divide_action_hidden_state(out)
        log_prob_action, log_prob_next_hidden_state = self.divide_action_hidden_state(log_prob)
        log_prob_action = log_prob_action.sum(dim=1)
        log_prob_next_hidden_state = log_prob_next_hidden_state.sum(dim=1)

        log_prob_action -= torch.log(1. - action.pow(2) + self._eps_log_prob).sum(dim=1)
        log_prob_next_hidden_state -= torch.log(1. - next_hidden_state.pow(2) + self._eps_log_prob).sum(dim=1)

        return out_true, log_prob_action, log_prob_next_hidden_state

    def compute_log_prob_for_action_hidden_state_given_dist_t(self, dist, action, hidden_state):

        action_hidden_state = torch.concat([action, hidden_state], dim=1)
        action_hidden_state_raw = torch.arctanh(((action_hidden_state - self._central_a) / self._delta_a) * (1 - 1e-7))
        log_prob = dist.log_prob(action_hidden_state_raw)

        log_prob_action, log_prob_next_hidden_state = self.divide_action_hidden_state(log_prob)
        log_prob_action = log_prob_action.sum(dim=1)
        log_prob_next_hidden_state = log_prob_next_hidden_state.sum(dim=1)

        log_prob_action -= torch.log(1. - action.pow(2) + self._eps_log_prob).sum(dim=1)
        log_prob_next_hidden_state -= torch.log(1. - hidden_state.pow(2) + self._eps_log_prob).sum(dim=1)

        return log_prob_action, log_prob_next_hidden_state

    def divide_action_hidden_state(self, out):
        return out[:, :self._action_dim], out[:, self._action_dim:]

    def distribution(self, state, prev_a, lengths):

        mu, log_sigma = self._approximator.predict(state, prev_a, lengths, output_tensor=True)

        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        return torch.distributions.Normal(mu, log_sigma.exp())

    def get_dist(self, state, prev_a):

        batch_size = state.shape[0]

        state = state.reshape(batch_size, 1, -1)
        prev_a = prev_a.reshape(batch_size, 1, -1)
        lengths = np.ones(batch_size, dtype=int)

        if type(state) == torch.Tensor:
            state = state.float()
        else:
            state = state.astype(np.float32)

        return self.distribution(state, prev_a.astype(np.float32), lengths)

    def sample_squashed_action_hidden_state(self, dist):

        out_raw = dist.rsample()
        out = torch.tanh(out_raw)
        out_true = out * self._delta_a + self._central_a

        action, next_hidden_state = self.divide_action_hidden_state(out_true)
        return action, next_hidden_state

    def entropy(self, state, prev_a, lengths=None):

        if len(state.shape) == 3:
            assert lengths is not None
            assert len(prev_a.shape) == 3

        if lengths is None:
            lengths = torch.ones(state.shape[0],
                                 dtype=torch.long)  # we use the same networks as for bptt_pI, but they require sequences, so we pass sequences with lengths 1

        return torch.mean(self.distribution(state, prev_a, lengths).entropy()).detach().cpu().numpy().item()

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        self._approximator.set_weights(weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        return self._approximator.get_weights()

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return self._approximator.model.network.parameters()


class SACRecurrentPolicy_Hybrid(SACRecurrentPolicy):

    def reset(self):
        if self._init_hidden_state is not None:
            self._last_hidden_state = deepcopy(self._init_hidden_state)
            self._last_hidden_state_deterministic = deepcopy(self._init_hidden_state)
        else:
            self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)
            self._last_hidden_state_deterministic = np.zeros((1, self._hidden_state_dim), dtype=np.float32)

    def draw_action(self, state):
        # WARNING: This function is not supposed to be called except from the cpg!
        if self._state_mask is not None:
            state = state[self._state_mask]
        state = np.atleast_2d(state)

        assert len(state) == 1 # because it is only meant to be called from the cpg at each step.
        # replace the stochastic last hidden state with the deterministic one
        env_state = state[:, :-self._hidden_state_dim]
        state = np.concatenate([env_state, deepcopy(self._last_hidden_state_deterministic)], axis=1)

        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        out, dist = self.compute_action_hidden_and_log_prob_t(state, prev_a,
                                                        compute_log_prob=False, return_dist=True)
        out = out.detach().cpu().numpy()
        action, next_hidden_state = self.divide_action_hidden_state(out)
        action = np.atleast_1d(np.squeeze(action))
        next_hidden_state = np.atleast_1d(np.squeeze(next_hidden_state))
        _, next_hidden_state_deterministic = self.divide_action_hidden_state(dist.mean.detach().cpu().numpy())
        self._last_action = deepcopy(action)
        self._last_hidden_state = deepcopy(next_hidden_state)
        self._last_hidden_state_deterministic = deepcopy(next_hidden_state_deterministic)
        return action


class SACBPTTPolicy(Policy):

    def __init__(self, approximator, hidden_state_dim, action_dim, lows, highs,
                 log_std_min, log_std_max, state_mask=None, init_hidden_state=None):

        self.use_mean = False  # if true the mean action is taken instead of sampling from Gaussian
        self._approximator = approximator
        self._output_shape = self._approximator.model.network.output_shape
        self._half_out_shape = self._output_shape // 2
        assert type(self._output_shape) == int, "Output shape needs to be an integer."
        assert 2 * self._half_out_shape == self._output_shape, "Output shape needs to be an even number."

        self._delta_a = to_float_tensor(.5 * (highs - lows), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (highs + lows), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._approximator.model.use_cuda

        self._hidden_state_dim = hidden_state_dim
        self._action_dim = action_dim
        self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)
        self._last_action = np.zeros(self._action_dim, dtype=np.float32)
        self._init_hidden_state = init_hidden_state
        self._state_mask = state_mask
        self.reset()

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive',
            _output_shape='primitive',
            _half_out_shape='primitive',
            _hidden_state_dim='numpy',
            _action_dim='numpy',
            _last_hidden_state='numpy',
            _last_action='numpy',
            _init_hidden_state='numpy',
            _state_mask='numpy',
        )

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def reset(self):
        if self._init_hidden_state is not None:
            self._last_hidden_state = deepcopy(self._init_hidden_state)
        else:
            self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):

        if self._state_mask is not None:
            state = state[self._state_mask]

        state = np.atleast_2d(state)
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        action, next_hidden_state = self.compute_action_hidden_and_log_prob_t(state, prev_a,
                                                                              compute_log_prob=False,
                                                                              return_hidden_state=True)
        action = np.atleast_1d(np.squeeze(action.detach().cpu().numpy()))
        next_hidden_state = np.squeeze(next_hidden_state.detach().cpu().numpy())
        self._last_action = deepcopy(action)
        self._last_hidden_state = deepcopy(next_hidden_state)
        return action

    def compute_action_hidden_and_log_prob(self, state, prev_a, lengths=None,
                                           compute_log_prob=True,return_hidden_state=False):
        out, log_prob = self.compute_action_hidden_and_log_prob_t(state, prev_a, lengths,
                                                                  compute_log_prob, return_hidden_state)
        return out.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_hidden_and_log_prob_t(self, state, prev_a, lengths=None, compute_log_prob=True,
                                             return_hidden_state=False):

        batch_size = state.shape[0]
        if len(state.shape) == 3:
            # we are using bptt_pI, so we need sequences
            assert lengths is not None
            assert len(prev_a.shape) == 3
        else:
            # we are not using bptt_pI, but we extend the dims to use the same bptt_pI network with sequence lengths 1
            state = state.reshape(batch_size, 1, -1)
            prev_a = prev_a.reshape(batch_size, 1, -1)
            lengths = np.ones(batch_size, dtype=int)

        if type(state) == np.ndarray:
            state = state.astype(np.float32)
        if type(prev_a) == np.ndarray:
            prev_a = prev_a.astype(np.float32)

        dist, next_hidden_state = self.distribution(state, prev_a, lengths)
        out_raw = dist.rsample()
        out = torch.tanh(out_raw)
        out_true = out * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(out_raw).sum(dim=1)
            log_prob -= torch.log(1. - out.pow(2) + self._eps_log_prob).sum(dim=1)
            if return_hidden_state:
                return out_true, log_prob, next_hidden_state
            else:
                return out_true, log_prob,
        else:
            if return_hidden_state:
                return out_true, next_hidden_state
            else:
                return out_true

    def divide_action_hidden_state(self, out):
        return out[:, :self._action_dim], out[:, self._action_dim:]

    def distribution(self, state, prev_a, lengths):

        mu, log_sigma, next_hidden  = self._approximator.predict(state, prev_a, lengths, output_tensor=True)

        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        return torch.distributions.Normal(mu, log_sigma.exp()), next_hidden

    def entropy(self, state, prev_a, lengths=None):

        if len(state.shape) == 3:
            assert lengths is not None
            assert len(prev_a.shape) == 3

        if lengths is None:
            lengths = torch.ones(state.shape[0],
                                 dtype=torch.long)  # we use the same networks as for bptt_pI, but they require sequences, so we pass sequences with lengths 1

        return torch.mean(self.distribution(state, prev_a, lengths).entropy()).detach().cpu().numpy().item()

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def use_cuda(self):
        return self._approximator.model.use_cuda

    def parameters(self):
        return self._approximator.model.network.parameters()


class SACBPTTPolicy_preprocess(SACBPTTPolicy):

    def __init__(self, state_mask, **kwargs):

        super().__init__(**kwargs)
        self._state_mask = state_mask

        self._add_save_attr(
            _state_mask='primitive'
        )

    def draw_action(self, state):
        state = state[self._state_mask]
        return super().draw_action(state)


class SACWindowPolicy(SACBPTTPolicy):

    def __init__(self, window_length, **kwargs):

        self._window_length = window_length
        super(SACWindowPolicy, self).__init__(**kwargs)

    def draw_action(self, state):
        state = np.atleast_2d(state[self._state_mask])
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        self.add_to_window(state, prev_a)
        current_window_obs = np.expand_dims(self._window_obs, axis=0)
        current_window_act = np.expand_dims(self._window_act, axis=0)
        lengths = np.array([1, self._window_length])
        action, next_hidden_state = self.compute_action_hidden_and_log_prob(current_window_obs,
                                                                              current_window_act,
                                                                              lengths=lengths,
                                                                              compute_log_prob=False,
                                                                              return_hidden_state=True)

        self._last_action = deepcopy(np.squeeze(action))
        self._last_hidden_state = deepcopy(np.squeeze(next_hidden_state))
        return np.squeeze(action)

    def reset(self):
        super(SACWindowPolicy, self).reset()

        self._window_obs = np.zeros((self._window_length, np.sum(self._state_mask)))
        self._window_act = np.zeros((self._window_length, self._action_dim))
        self._curr_wind_ind = 0

    def distribution(self, state, prev_a, lengths):
        mu, log_sigma = self._approximator.predict(state, prev_a, lengths, output_tensor=True)

        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        # return distribution and zeros as hidden state (the latter is not needed here)
        return torch.distributions.Normal(mu, log_sigma.exp()), torch.zeros((len(mu), self._hidden_state_dim))

    def add_to_window(self, state, prev_action):
        if self._curr_wind_ind == self._window_length:
            self._window_obs = np.delete(self._window_obs, 0, 0)
            self._window_obs = np.concatenate([self._window_obs, state])
            self._window_act = np.delete(self._window_act, 0, 0)
            self._window_act = np.concatenate([self._window_act, prev_action])
        else:
            self._window_obs[self._curr_wind_ind] = state[0]
            self._window_act[self._curr_wind_ind] = prev_action[0]
            self._curr_wind_ind += 1


class SACPolicy_QState(Policy):

    def __init__(self, approximator, hidden_state_dim, action_dim, lows, highs,
                 log_std_min, log_std_max, critic_approximator, init_hidden_state=None):

        self.use_mean = False  # if true the mean action is taken instead of sampling from Gaussian
        self._approximator = approximator
        self._output_shape = self._approximator.model.network.output_shape
        self._half_out_shape = self._output_shape // 2
        assert type(self._output_shape) == int, "Output shape needs to be an integer."
        assert 2 * self._half_out_shape == self._output_shape, "Output shape needs to be an even number."
        self._critic_approximator = critic_approximator

        self._delta_a = to_float_tensor(.5 * (highs - lows), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (highs + lows), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._approximator.model.use_cuda

        self._hidden_state_dim = hidden_state_dim
        self._action_dim = action_dim
        self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)
        self._last_action = np.zeros(self._action_dim, dtype=np.float32)
        self._init_hidden_state = init_hidden_state
        self.reset()

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive',
            _output_shape='primitive',
            _half_out_shape='primitive',
            _hidden_state_dim='numpy',
            _action_dim='numpy',
            _last_hidden_state='numpy',
            _last_action='numpy',
            _init_hidden_state='numpy',
            _critic_approximator='torch'
        )

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def reset(self):
        if self._init_hidden_state is not None:
            self._last_hidden_state = deepcopy(self._init_hidden_state)
        else:
            self._last_hidden_state = np.zeros(self._hidden_state_dim, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        state = np.atleast_2d(state)
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        action, next_hidden_state = self.compute_action_hidden_and_log_prob_t(state, prev_a,
                                                                              compute_log_prob=False,
                                                                              return_hidden_state=True)
        action = np.atleast_1d(np.squeeze(action.detach().cpu().numpy()))
        next_hidden_state = np.squeeze(next_hidden_state)
        self._last_action = deepcopy(action)
        self._last_hidden_state = deepcopy(next_hidden_state)
        return action

    def compute_action_hidden_and_log_prob(self, state, prev_a, lengths=None):
        out, log_prob = self.compute_action_hidden_and_log_prob_t(state, prev_a, lengths)
        return out.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_hidden_and_log_prob_t(self, state, prev_a, lengths=None, compute_log_prob=True,
                                             return_hidden_state=False):

        batch_size = state.shape[0]
        if len(state.shape) == 3:
            # we are using bptt_pI, so we need sequences
            raise ValueError("Sequences are not allowed in this policy.")
        if return_hidden_state:
            dist, next_hidden_state = self.distribution(state.astype(np.float32), prev_a.astype(np.float32), return_hidden_state)
        else:
            if type(state) == torch.Tensor:
                state = state.type(torch.FloatTensor)
            else:
                state = state.astype(np.float32)
            dist = self.distribution(state, prev_a.astype(np.float32), return_hidden_state)

        out_raw = dist.rsample()
        out = torch.tanh(out_raw)
        out_true = out * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(out_raw).sum(dim=1)
            log_prob -= torch.log(1. - out.pow(2) + self._eps_log_prob).sum(dim=1)
            if return_hidden_state:
                return out_true, log_prob, next_hidden_state
            else:
                return out_true, log_prob,
        else:
            if return_hidden_state:
                return out_true, next_hidden_state
            else:
                return out_true

    def divide_action_hidden_state(self, out):
        return out[:, :self._action_dim], out[:, self._action_dim:]

    def distribution(self, state, prev_a, return_hidden_state):

        mu_log_sigma  = self._approximator.predict(state, prev_a, output_tensor=True)

        mu = mu_log_sigma[:, :self._half_out_shape]
        log_sigma = mu_log_sigma[:, self._half_out_shape:]

        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        if return_hidden_state:
            # we need the hidden state of the critic, therefore we need a random action,
            # which is not used for calculation of the hiddens state (but only the q-value, which is not needed here)
            batch_size = state.shape[0]
            lengths = np.ones(batch_size, dtype=int)
            state = state.reshape(batch_size, 1, -1)
            prev_a = prev_a.reshape(batch_size, 1, -1)
            zero_action = np.zeros((batch_size, self._action_dim))
            _, next_hidden = self._critic_approximator(state, zero_action, prev_a, lengths,
                                                       use_hidden_states=True, return_hidden_states=True)

            return torch.distributions.Normal(mu, log_sigma.exp()), next_hidden
        else:
            return torch.distributions.Normal(mu, log_sigma.exp())

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def use_cuda(self):
        return self._approximator.model.use_cuda

    def parameters(self):
        return self._approximator.model.network.parameters()


class StatefulDiagonalGaussianPolicy(DiagonalGaussianPolicy):

    def __init__(self, mu, logstd, dim_action, dim_hidden_state, learnable_std=True):

        self.dim_action = dim_action
        self.dim_hidden_state = dim_hidden_state
        self._last_hidden_state = np.zeros(self.dim_hidden_state, dtype=np.float32)
        self._last_action = np.zeros(self.dim_action, dtype=np.float32)

        super().__init__(mu, logstd, learnable_std=learnable_std)

        self._add_save_attr(
            dim_action='primitive',
            dim_hidden_state='primitive',
            _last_hidden_state='numpy',
            _last_action='numpy')

    def get_last_hidden_state(self):
        return deepcopy(self._last_hidden_state)

    def get_last_action(self):
        return deepcopy(self._last_action)

    def draw_action(self, state):
        state = np.atleast_2d(state)
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        mu, sigma = self._compute_multivariate_gaussian(state.astype(np.float32), prev_a)[:2]
        outs = np.random.multivariate_normal(mu, sigma).astype(np.float32)
        action, next_hidden_state = self.divide_action_hidden_state(outs)
        self._last_action = deepcopy(action)
        self._last_hidden_state = deepcopy(next_hidden_state)
        return action

    def _compute_multivariate_gaussian(self, state, prev_a, lengths=None):

        mus = self._compute_mean(state, prev_a, lengths)

        sigma = np.square(np.exp(self._logstd))

        return mus, np.diag(sigma), np.diag(1. / sigma)

    def _compute_multivariate_gaussian_batch(self, state, prev_a, lengths=None):

        mus = self._compute_mean(state, prev_a, lengths)

        sigma = np.square(np.exp(self._logstd))
        n_mus = len(mus)
        return mus, np.tile(np.diag(sigma), (n_mus, 1, 1)), np.tile(np.diag(1. / sigma), (n_mus, 1, 1))

    def _compute_mean(self, state, prev_a, lengths=None):

        batch_size = state.shape[0]
        if len(state.shape) == 3:
            # we are using bptt_pI, so we need sequences
            assert lengths is not None
            assert len(prev_a.shape) == 3
        else:
            # we are not using bptt_pI, but we extend the dims to use the same bptt_pI network with sequence lengths 1
            state = state.reshape(batch_size, 1, -1)
            prev_a = prev_a.reshape(batch_size, 1, -1)
            lengths = np.ones(batch_size, dtype=int)

        mus = self._approximator.predict(state, prev_a, lengths, **self._predict_params)

        return np.squeeze(mus)

    def divide_action_hidden_state(self, out):
        return out[:self.dim_action], out[self.dim_action:]

    def divide_state_to_env_hidden_batch(self, state):
        if len(state.shape) == 2:
            return state[:, -self.dim_hidden_state], state[:, -self.dim_hidden_state:]
        elif len(state.shape) == 3:
            return state[:, :, -self.dim_hidden_state], state[:, :,  -self.dim_hidden_state:]
        else:
            raise ValueError()

    def diff_log_batch(self, states, actions, next_states, prev_action, lengths=None):
        states = states.astype(np.float32)
        actions = actions.astype(np.float32)
        _, next_hidden_states = self.divide_state_to_env_hidden_batch(next_states)
        if len(next_hidden_states.shape) == 3:
            next_hidden_states = next_hidden_states[:, -1, :]
        actions_next_hidden = np.concatenate([actions, next_hidden_states], axis=1)

        mus, _, inv_sigmas = self._compute_multivariate_gaussian_batch(states, prev_action, lengths)

        deltas = actions_next_hidden - mus

        # compute the weights for reduction
        inv_sigmas_T = np.transpose(inv_sigmas, axes=[0, 2, 1])
        deltas_T = np.expand_dims(deltas, 2)
        w = .5 * np.matmul(inv_sigmas + inv_sigmas_T, deltas_T)

        # Compute mean derivative
        g_mus = self._approximator.diff_batch(states, prev_action, lengths=lengths, reduction_weights=w)

        # Compute standard deviation derivative
        if self._learnable_std:
            g_sigma = -1 + deltas ** 2 / np.exp(self._logstd) ** 3
            g_sigma = np.sum(g_sigma, 0)
        else:
            g_sigma = np.zeros_like(self._logstd)

        return np.concatenate((g_mus, g_sigma), axis=0)


class BPTTDiagonalGaussianPolicy(StatefulDiagonalGaussianPolicy):

    def draw_action(self, state):
        state = np.atleast_2d(state)
        prev_a = self.get_last_action()
        prev_a = np.atleast_2d(prev_a)
        mu, sigma = self._compute_multivariate_gaussian(state.astype(np.float32), prev_a)[:2]
        outs = mu
        ind_log_std = len(self._logstd)
        action = np.random.multivariate_normal(np.atleast_1d(mu[0:ind_log_std]), sigma).astype(np.float32)
        next_hidden_state = outs[ind_log_std:]
        self._last_action = deepcopy(action)
        self._last_hidden_state = deepcopy(next_hidden_state)
        return action

    def diff_log_batch(self, states, actions, next_states, prev_action, lengths=None):
        states = states.astype(np.float32)
        actions = actions.astype(np.float32)

        ind_actions = len(self._logstd)

        mus, _, inv_sigmas = self._compute_multivariate_gaussian_batch(states, prev_action, lengths)

        deltas = actions - mus[:, :ind_actions]

        # compute the weights for reduction
        inv_sigmas_T = np.transpose(inv_sigmas, axes=[0, 2, 1])
        deltas_T = np.expand_dims(deltas, 2)
        w = .5 * np.matmul(inv_sigmas + inv_sigmas_T, deltas_T)

        w_hidden_state = np.tile(np.zeros((w.shape[0], 1, 1)), (1, self.dim_hidden_state, 1))

        w = np.concatenate([w, w_hidden_state], axis=1)

        # Compute mean derivative
        g_mus = self._approximator.diff_batch(states, prev_action, lengths=lengths, reduction_weights=w)

        # Compute standard deviation derivative
        if self._learnable_std:
            g_sigma = -1 + deltas ** 2 / np.exp(self._logstd) ** 3
            g_sigma = np.sum(g_sigma, 0)
        else:
            g_sigma = np.zeros_like(self._logstd)

        return np.concatenate((g_mus, g_sigma), axis=0)
