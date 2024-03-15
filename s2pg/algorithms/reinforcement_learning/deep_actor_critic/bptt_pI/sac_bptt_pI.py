import warnings

import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from s2pg.policy import SACBPTTPolicy
from s2pg.utils.replay_memory import SequenceReplayMemory

from copy import deepcopy
from itertools import chain


class SAC_BPTT_pI(DeepAC):
    """
    Soft Actor-Critic algorithm.
    Implementation with a recurrent policy and a standard MLP critic. This is supposed to be used
    in the privileged information (pI) setting, where the critic has the full state information, and
    the policy only receives observations.

    Non-Recurrent version first appeared in:
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """
    def __init__(self, mdp_info, actor_params,
                 actor_optimizer, critic_params, batch_size, sw,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, hidden_state_dim, action_dim, critic_state_mask=None, policy_state_mask=None,
                 policy_type_params=None, init_alpha=0.1, logging_iter=1000, log_std_min=-20, log_std_max=2,
                 target_entropy=None, critic_fit_params=None, truncation_length=5, policy_type=SACBPTTPolicy):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigma
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape[0]).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._truncation_length = truncation_length
        self._replay_memory = SequenceReplayMemory(truncation_length, initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        actor_approximator = Regressor(TorchApproximator,
                                       **actor_params)

        policy_type_params = dict() if policy_type_params is None else policy_type_params
        policy = policy_type(approximator=actor_approximator,
                             hidden_state_dim=hidden_state_dim,
                             action_dim=action_dim,
                             lows=mdp_info.action_space.low,
                             highs=mdp_info.action_space.high,
                             log_std_min=log_std_min,
                             log_std_max=log_std_max,
                             **policy_type_params)

        self._hidden_state_dim = hidden_state_dim

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_approximator.model.network.parameters())

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._logging_iter = logging_iter
        self._iter = 0

        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)
        self._critic_state_mask = critic_state_mask if critic_state_mask is not None else\
            np.concatenate([np.ones(mdp_info.observation_space.shape[0], dtype=bool),
                            np.zeros(hidden_state_dim, dtype=bool)])

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch',
            _logging_iter='primitive',
            _iter='primitive',
            _truncation_lengths='primitive',
            _critic_state_mask='numpy',
            _policy_state_mask='numpy',
            _Q_fits='primitive'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        # add the preprocessor to append the hidden state to the environment state
        self._preprocessors.append(self.append_hidden_state)

    def add_preprocessor(self, preprocessor):
        # for now disable the preprocessor to ensure that appending the hidden state is always done at last
        raise AttributeError("This agent current does not support preprocessors.")

    def add_hidden_state_preprocessor(self):
        if len(self._preprocessors) == 0:  # only the hidden state preprocessor is allowed for now, which is why we can check the length
            self._preprocessors.append(self.append_hidden_state)
        else:
            warnings.warn("Hidden state preprocessor already included, and will be not added twice.")

    def append_hidden_state(self, x):
        # get latest hidden state
        hidden_state = self.policy.get_last_hidden_state()
        return np.concatenate([x, hidden_state])

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, prev_action, lengths, action_seq =\
                self._replay_memory.get(self._batch_size())

            rel_indices = lengths.reshape(-1, 1, 1) - 1
            last_state = np.squeeze(np.take_along_axis(state, rel_indices, axis=1), axis=1)
            last_next_state = np.squeeze(np.take_along_axis(next_state, rel_indices, axis=1), axis=1)

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = \
                    self.policy.compute_action_hidden_and_log_prob_t(state[:, :, self._policy_state_mask],
                                                                     prev_action,
                                                                     lengths)
                loss = self._loss(last_state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())

            if self._iter % self._logging_iter == 0 and self._replay_memory.size > self._warmup_transitions():

                grads = []
                for param in self.policy._approximator.model.network.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                norm = grads.norm(dim=0, p=2)

                self._sw.add_scalar("Actor/Joint Entropy ", np.mean(-log_prob.detach().cpu().numpy()), self._iter)
                self._sw.add_scalar("Actor/Alpha Param ", np.mean(self._alpha.detach().cpu().numpy()), self._iter)
                self._sw.add_scalar("Gradients/Norm2 Gradient Q wrt. Pi-parameter", torch.mean(norm), self._iter)

            rel_indices = lengths.reshape(-1, 1, 1) - 1
            last_state = np.squeeze(np.take_along_axis(state, rel_indices, axis=1), axis=1)
            last_next_state = np.squeeze(np.take_along_axis(next_state, rel_indices, axis=1), axis=1)

            q_next = self._next_q(last_next_state, next_state, action_seq, absorbing, lengths)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(last_state[:, self._critic_state_mask], action,  q,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

            if self._iter % self._logging_iter == 0:
                self._sw.add_scalar("Q-Value", np.mean(q), self._iter)

        self._iter += 1

    def _loss(self, state, action, log_prob):
        q_0 = self._critic_approximator(state[:, self._critic_state_mask], action,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state[:, self._critic_state_mask], action,
                                        output_tensor=True, idx=1)
        q = torch.min(q_0, q_1)
        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, next_state_seq, action_seq, absorbing, lengths):
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
        next_action, log_prob_next =\
            self.policy.compute_action_hidden_and_log_prob(next_state_seq[:, :, self._policy_state_mask],
                                                           action_seq,
                                                           lengths)

        q = self._target_critic_approximator.predict(next_state[:, self._critic_state_mask], next_action, prediction = 'min')


        q -= self._alpha_np * log_prob_next

        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
