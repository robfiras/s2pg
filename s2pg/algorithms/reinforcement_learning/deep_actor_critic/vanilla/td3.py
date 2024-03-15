import numpy as np

from s2pg.algorithms.reinforcement_learning.deep_actor_critic.vanilla.ddpg import DDPG
from mushroom_rl.policy import Policy
from mushroom_rl.policy import ParametricPolicy
from mushroom_rl.utils.parameters import to_parameter



class ClippedGaussianPolicy(ParametricPolicy):
    """
    Clipped Gaussian policy, as used in:

    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    This is a non-differentiable policy for continuous action spaces.
    The policy samples an action in every state following a gaussian
    distribution, where the mean is computed in the state and the covariance
    matrix is fixed. The action is then clipped using the given action range.
    This policy is not a truncated Gaussian, as it simply clips the action
    if the value is bigger than the boundaries. Thus, the non-differentiability.

    """
    def __init__(self, mu, sigma, low, high, state_mask=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the
                state;
            sigma (np.ndarray): a square positive definite matrix representing
                the covariance matrix. The size of this matrix must be n x n,
                where n is the action dimensionality;
            low (np.ndarray): a vector containing the minimum action for each
                component;
            high (np.ndarray): a vector containing the maximum action for each
                component.

        """
        self._approximator = mu
        self._predict_params = dict()
        self._sigma = sigma
        self._low = low
        self._high = high
        self.use_mean = False
        self._state_mask = state_mask

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _sigma='numpy',
            _low='numpy',
            _high='numpy',
            use_mean='primitive'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        if self._state_mask is not None:
            state = state[self._state_mask]

        if self.use_mean:
            return np.reshape(self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params), -1)
        else:
            mu = np.reshape(self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params), -1)

            action_raw = np.random.multivariate_normal(mu, self._sigma)

        return np.clip(action_raw, self._low, self._high)

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size



class TD3(DDPG):
    """
    Twin Delayed DDPG algorithm.
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    """
    def __init__(self, mdp_info, policy_params, actor_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay=2,
                 sw=None, logging_iter=10, policy_state_mask=None,
                 noise_std=.2, noise_clip=.5, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std ([float, Parameter], .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip ([float, Parameter], .5): maximum absolute value for policy smoothing
                noise;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._noise_std = to_parameter(noise_std)
        self._noise_clip = to_parameter(noise_clip)

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2

        self._add_save_attr(
            _noise_std='mushroom',
            _noise_clip='mushroom'
        )

        policy_class = ClippedGaussianPolicy

        super().__init__(mdp_info, policy_class, policy_params,  actor_params,
                         actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, tau, policy_state_mask,
                         policy_delay, sw, logging_iter, critic_fit_params)

    def _loss(self, state):
        action = self._actor_approximator(state[:, self._policy_state_mask], output_tensor=True, **self._actor_predict_params)
        q = self._critic_approximator(state, action, idx=0, output_tensor=True, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self._target_actor_approximator(next_state[:, self._policy_state_mask], **self._actor_predict_params)

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        eps = np.random.normal(scale=self._noise_std(), size=a.shape)
        eps_clipped = np.clip(eps, -self._noise_clip(), self._noise_clip.get_value())
        a_smoothed = np.clip(a + eps_clipped, low, high)

        q = self._target_critic_approximator.predict(next_state, a_smoothed,
                                                     prediction='min', **self._critic_predict_params)
        q *= 1 - absorbing

        return q
