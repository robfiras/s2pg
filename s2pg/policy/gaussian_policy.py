import numpy as np

from mushroom_rl.policy import ParametricPolicy
from mushroom_rl.approximators import Regressor
from mushroom_rl.policy.torch_policy import GaussianTorchPolicy

from scipy.stats import multivariate_normal


class AbstractGaussianPolicy(ParametricPolicy):
    """
    Abstract class of Gaussian policies.

    """

    def __init__(self,  **kwargs):
        self.deterministic = False
        super().__init__(**kwargs)
        self._add_save_attr(
            determinisitc='primitive')

    def __call__(self, state, action):
        mu, sigma = self._compute_multivariate_gaussian(state)[:2]
        if not self.deterministic:
            return multivariate_normal.pdf(action, mu, sigma)
        else:
            return mu

    def draw_action(self, state):
        mu, sigma = self._compute_multivariate_gaussian(state.astype(np.float32))[:2]
        action = np.random.multivariate_normal(mu, sigma).astype(np.float32)
        return action


class DiagonalGaussianPolicy(AbstractGaussianPolicy):
    """
    Gaussian policy with learnable standard deviation.
    The Covariance matrix is
    constrained to be a diagonal matrix, where the diagonal is the squared
    standard deviation vector.
    This is a differentiable policy for continuous action spaces.
    This policy is similar to the gaussian policy, but the weights includes
    also the standard deviation.

    """
    def __init__(self, mu, logstd, learnable_std=True):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            std (np.ndarray): a vector of standard deviations. The length of
                this vector must be equal to the action dimensionality.

        """
        self._approximator = mu
        self._predict_params = dict()
        self._logstd = logstd.astype(np.float32)
        self._learnable_std = learnable_std

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _learnanle_std='primitive',
            _logstd='numpy'
        )

    def set_std(self, logstd):
        """
        Setter.

        Args:
            std (np.ndarray): the new standard deviation. Must be a square
                positive definite matrix.

        """
        self._logstd = logstd

    def diff_log(self, state, action):
        mu, _, inv_sigma = self._compute_multivariate_gaussian(state)

        delta = action - mu

        # Compute mean derivative
        j_mu = self._approximator.diff(state)

        if len(j_mu.shape) == 1:
            j_mu = np.expand_dims(j_mu, axis=1)

        g_mu = .5 * j_mu.dot(inv_sigma + inv_sigma.T).dot(delta.T)

        # Compute standard deviation derivative
        if self._learnable_std:
            g_sigma = -1 + delta ** 2 / np.exp(self._logstd) ** 3
        else:
            g_sigma = np.zeros_like(self._logstd)

        return np.concatenate((g_mu, g_sigma), axis=0)

    def diff_log_batch(self, states, actions):
        states = states.astype(np.float32)
        actions = actions.astype(np.float32)

        mus, _, inv_sigmas = self._compute_multivariate_gaussian_batch(states)

        deltas = actions - mus

        # compute the weights for reduction
        inv_sigmas_T = np.transpose(inv_sigmas, axes=[0, 2, 1])
        deltas_T = np.expand_dims(deltas, 2)
        w = .5 * np.matmul(inv_sigmas + inv_sigmas_T, deltas_T)

        # Compute mean derivative
        g_mus = self._approximator.model.diff_batch(states, reduction_weights=w)

        # Compute standard deviation derivative
        if self._learnable_std:
            g_sigma = -1 + deltas ** 2 / np.exp(self._logstd) ** 3
            g_sigma = np.sum(g_sigma, 0)
        else:
            g_sigma = np.zeros_like(self._logstd)

        return np.concatenate((g_mus, g_sigma), axis=0)

    def set_weights(self, weights):
        self._approximator.set_weights(
            weights[0:self._approximator.weights_size])
        self._logstd = weights[self._approximator.weights_size:]

    def get_weights(self):
        return np.concatenate((self._approximator.get_weights(), self._logstd),
                              axis=0)

    def get_weight_names(self):
        return self._approximator.model.get_weight_names() + ["std_%d" % i for i, _ in enumerate(self._logstd)]

    @property
    def weights_size(self):
        return self._approximator.weights_size + self._logstd.size

    def entropy(self):
        var = np.square(np.exp(self._logstd))
        return len(var) * (0.5 + 0.5 * np.log(2 * np.pi)) + 0.5 * np.sum(var)

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params), -1)

        sigma = np.square(np.exp(self._logstd))

        return mu, np.diag(sigma), np.diag(1. / sigma)

    def _compute_multivariate_gaussian_batch(self, states):

        mus = self._approximator.predict(states, **self._predict_params)

        sigma = np.square(np.exp(self._logstd))
        n_mus = len(mus)
        return mus, np.tile(np.diag(sigma), (n_mus, 1, 1)), np.tile(np.diag(1. / sigma), (n_mus, 1, 1))


class PPOBasePolicy(GaussianTorchPolicy):

    def __init__(self, state_mask=None, **kwargs):

        self._state_mask = state_mask
        super(PPOBasePolicy, self).__init__(**kwargs)

    def draw_action(self, state):

        if self._state_mask is not None:
            state = state[self._state_mask]

        return super(PPOBasePolicy, self).draw_action(state)
