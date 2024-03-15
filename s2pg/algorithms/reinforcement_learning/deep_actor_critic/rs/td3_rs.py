import numpy as np
import torch

from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from .ddpg_rs import DDPG_RS


class TD3_RS(DDPG_RS):
    """
    Twin Delayed DDPG algorithm formulated with the stochastic stateful policy gradient.
    Implementation with a recurrent stochastic policy (hence RS) and a critic that takes the policy states
    into account. This can be used in settings with and without privileged information, even though without
    the privileged information the performance is poor (see paper).

    Non-Recurrent version first appeared in:
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    """

    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, tau_actor=None, policy_delay=2,
                 noise_std=.2, noise_clip=.5, noise_hidden_std=.2, noise_hidden_clip=.5,
                 critic_fit_params=None, loss_mult_hidden_state=1.0, Q_clipping=False,
                 policy_state_mask=None, critic_state_mask=None, **kwargs):

        self._noise_std = to_parameter(noise_std)
        self._noise_clip = to_parameter(noise_clip)
        self._noise_hidden_std = to_parameter(noise_hidden_std)
        self._noise_hidden_clip = to_parameter(noise_hidden_clip)
        self._loss_mult_hidden_state = loss_mult_hidden_state

        if tau_actor is None:
            tau_actor = tau

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2

        self._add_save_attr(
            _noise_std='mushroom',
            _noise_clip='mushroom'
        )

        super().__init__(mdp_info=mdp_info, policy_class=policy_class,
                         policy_params=policy_params,
                         actor_params=actor_params,
                         actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                         initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                         tau_actor=tau_actor, Q_clipping=Q_clipping,
                         policy_state_mask=policy_state_mask, critic_state_mask=critic_state_mask,
                         policy_delay=policy_delay, critic_fit_params=critic_fit_params, **kwargs)

    def _loss(self, state, prev_action):
        action, next_hidden_state = self._actor_approximator.model.network(to_float_tensor(state[:, self._policy_state_mask]),
                                                                           to_float_tensor(prev_action))

        q = self._critic_approximator(state[:, self._critic_state_mask], action, next_hidden_state, prev_action,
                                      output_tensor=True,
                                      **self._critic_predict_params)

        return -q.mean(), next_hidden_state

    def _next_q(self, next_state, absorbing, prev_action):
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
        action, next_next_hidden_state = self._target_actor_approximator.model.network(to_float_tensor(next_state[:, self._policy_state_mask]),
                                                                                       to_float_tensor(prev_action))
        action = action.detach().cpu().numpy()
        next_next_hidden_state = next_next_hidden_state.detach().cpu().numpy()

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        low_hidden = self.policy.low_hidden
        high_hidden = self.policy.high_hidden
        action_eps = np.random.normal(scale=self._noise_std(), size=action.shape)
        action_eps_clipped = np.clip(action_eps, -self._noise_clip(), self._noise_clip())
        hidden_eps = np.random.normal(scale=self._noise_hidden_std(), size=next_next_hidden_state.shape)
        hidden_eps_clipped = np.clip(hidden_eps, -self._noise_hidden_clip(), self._noise_hidden_clip())
        a_smoothed = np.clip(action + action_eps_clipped, low, high)
        nhidden_smoothed = np.clip(next_next_hidden_state + hidden_eps_clipped, low_hidden, high_hidden)

        q = self._target_critic_approximator.predict(next_state[:, self._critic_state_mask], a_smoothed, nhidden_smoothed, prev_action,
                                                     prediction="min", **self._critic_predict_params)
        q *= 1 - absorbing

        return q