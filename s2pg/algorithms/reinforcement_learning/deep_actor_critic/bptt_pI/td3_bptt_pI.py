import numpy as np
import torch

from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from s2pg.algorithms.reinforcement_learning.deep_actor_critic.bptt_pI.ddpg_bptt_pI import DDPG_BPTT_pI


class TD3_BPTT_pI(DDPG_BPTT_pI):
    """
    Twin Delayed DDPG algorithm.
    Implementation with a recurrent policy and a standard MLP critic. This is supposed to be used
    in the privileged information (pI) setting, where the critic has the full state information, and
    the policy only receives observations.

    Non-Recurrent version first appeared in:
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, truncation_length, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay=2,
                 noise_std=.2, noise_clip=.5, critic_fit_params=None, **kwargs):

        self._noise_std = to_parameter(noise_std)
        self._noise_clip = to_parameter(noise_clip)

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] == 2)

        else:
            critic_params['n_models'] = 2

        self._add_save_attr(
            _noise_std='mushroom',
            _noise_clip='mushroom'
        )

        super().__init__(mdp_info=mdp_info, policy_class=policy_class,
                         policy_params=policy_params,
                         actor_params=actor_params, truncation_length=truncation_length,
                         actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                         initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                         policy_delay=policy_delay, critic_fit_params=critic_fit_params, **kwargs)

    def _loss(self, state, prev_action, lengths):
        action, _ = self._actor_approximator.model.network(to_float_tensor(state[:, :, self._policy_state_mask]),
                                                           to_float_tensor(prev_action),
                                                           torch.tensor(lengths))

        rel_indices = lengths.reshape(-1, 1, 1) - 1
        last_state = np.squeeze(np.take_along_axis(state, rel_indices, axis=1), axis=1)

        q = self._critic_approximator(last_state[:, self._critic_state_mask], action, output_tensor=True,
                                      **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, next_state, absorbing, prev_action, lengths):
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
        action, _ = self._target_actor_approximator.model.network(to_float_tensor(next_state[:, :, self._policy_state_mask]),
                                                                  to_float_tensor(prev_action),
                                                                  torch.tensor(lengths))
        action = action.detach().cpu().numpy()

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        action_eps = np.random.normal(scale=self._noise_std(), size=action.shape)
        action_eps_clipped = np.clip(action_eps, -self._noise_clip(), self._noise_clip())
        a_smoothed = np.clip(action + action_eps_clipped, low, high)

        rel_indices = lengths.reshape(-1, 1, 1) - 1
        last_next_state = np.squeeze(np.take_along_axis(next_state, rel_indices, axis=1), axis=1)

        q = self._target_critic_approximator.predict(last_next_state[:, self._critic_state_mask], a_smoothed,
                                                     **self._critic_predict_params)

        q *= 1 - absorbing

        return q
