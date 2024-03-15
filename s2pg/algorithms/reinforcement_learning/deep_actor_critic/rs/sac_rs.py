import warnings

import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from s2pg.policy import SACRecurrentPolicy
from s2pg.utils.replay_memory import ReplayMemoryPrevAction_with_return, ReplayMemoryPrevAction_with_returnv2

from copy import deepcopy
from itertools import chain


class SAC_RS(DeepAC):
    """
    Soft Actor-Critic algorithm formulated with the stochastic stateful policy gradient.
    Implementation with a recurrent stochastic policy (hence RS) and a critic that takes the policy states
    into account. This can be used in settings with and without privileged information, even though without
    the privileged information the performance is poor (see paper).

    Non-Recurrent version first appeared in:
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al. 2019.

    """
    def __init__(self, mdp_info, actor_params,
                 actor_optimizer, critic_params, batch_size, sw,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, lr_alpha_hidden, hidden_state_dim, action_dim,  init_alpha=0.1, init_alpha_hidden=0.1,
                 logging_iter=1000, log_std_min=-20, log_std_max=2, target_entropy=None, add_hidden_entropy=True,
                 critic_fit_params=None, alternating_training_interval_pol_hidden=None, critic_state_mask=None,
                 policy_state_mask=None, start_train_with="hidden_prob_kernel", policy_type=SACRecurrentPolicy,
                 policy_type_params=None, new_state_update_tau=0.0, Q_fits=1):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigm
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
        self._new_state_update_tau = to_parameter(new_state_update_tau)
        self._Q_fits = Q_fits
        self._add_hidden_entropy = add_hidden_entropy

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape[0]).astype(np.float32)
            self._target_entropy_hidden = -np.prod(hidden_state_dim).astype(np.float32)
        else:
            self._target_entropy = target_entropy
            self._target_entropy_hidden = target_entropy    #todo: maybe we want to add a specific target entropy for the hidden states

        self._replay_memory = ReplayMemoryPrevAction_with_returnv2(initial_size=initial_replay_size,
                                                                   max_size=max_replay_size)
        #self._replay_memory = ReplayMemoryPrevAction_with_return(mdp_info=mdp_info, initial_size=initial_replay_size,
        #                                                         max_size=max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)
        self._critic_state_mask = critic_state_mask if critic_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        actor_approximator = Regressor(TorchApproximator,
                                       **actor_params)

        lows = np.concatenate([mdp_info.action_space.low, -np.ones(hidden_state_dim)])
        highs = np.concatenate([mdp_info.action_space.high, np.ones(hidden_state_dim)])
        policy_type_params = dict() if policy_type_params is None else policy_type_params
        policy = policy_type(approximator=actor_approximator,
                             hidden_state_dim=hidden_state_dim,
                             action_dim=action_dim,
                             lows=lows,
                             highs=highs,
                             log_std_min=log_std_min,
                             log_std_max=log_std_max,
                             **policy_type_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32)
        self._log_alpha_hidden = torch.tensor(np.log(init_alpha_hidden), dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
            self._log_alpha_hidden = self._log_alpha_hidden.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()
            self._log_alpha_hidden.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)
        self._alpha_hidden_optim = optim.Adam([self._log_alpha_hidden], lr=lr_alpha_hidden)

        policy_parameters = chain(actor_approximator.model.network.parameters())

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._logging_iter = logging_iter
        self._iter = 0

        self._alternating_training_interval_pol_hidden = alternating_training_interval_pol_hidden
        self._current_turn = start_train_with

        self._hidden_state_dim = hidden_state_dim

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
            _alternating_training_interval_pol_hidden='primitive',
            _current_turn="primitive",
            _hidden_state_dim="primitive"
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
            state, action, reward, next_state, absorbing, _, prev_action, indices =\
                self._replay_memory.get(self._batch_size())

            if self._alternating_training_interval_pol_hidden is not None and \
                    self._iter % self._alternating_training_interval_pol_hidden == 0:
                if self._current_turn == "policy":
                    self._current_turn = "hidden_prob_kernel"
                else:
                    self._current_turn = "policy"

            if self._replay_memory.size > self._warmup_transitions():
                out_new, log_prob_action, log_prob_hidden =\
                    self.policy.compute_log_prob_separate_t(state[:, self._policy_state_mask], prev_action)
                loss = self._loss(state, prev_action, out_new, log_prob_action, log_prob_hidden)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob_action.detach())
                self._update_alpha_hidden(log_prob_hidden.detach())

            if self._iter % self._logging_iter == 0 and self._replay_memory.size > self._warmup_transitions() :

                grads = []
                for param in self.policy._approximator.model.network.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                norm = grads.norm(dim=0, p=2)

                self._sw.add_scalar("Actor/Entropy Action ", np.mean(-log_prob_action.detach().cpu().numpy()), self._iter)
                self._sw.add_scalar("Actor/Entropy Hidden State", np.mean(-log_prob_hidden.detach().cpu().numpy()), self._iter)
                self._sw.add_scalar("Actor/Alpha Param ", self._alpha_np, self._iter)
                self._sw.add_scalar("Actor/Alpha Hidden Param ", self._alpha_np_hidden, self._iter)
                self._sw.add_scalar("Gradients/Norm2 Gradient Q wrt. Pi-parameter", torch.mean(norm), self._iter)

            for i in range(self._Q_fits):

                q_next = self._next_q(next_state, action, absorbing)
                q = reward + self.mdp_info.gamma * q_next

                hidden_state = state[:, self._critic_state_mask][:, self.mdp_info.observation_space.shape[0]:]
                next_hidden_state = next_state[:, self._critic_state_mask][:, self.mdp_info.observation_space.shape[0]:]
                self._critic_approximator.fit(state[:, self._critic_state_mask], action, next_hidden_state,
                                              prev_action, q, **self._critic_fit_params)

                self._update_target(self._critic_approximator,
                                    self._target_critic_approximator)

            if self._iter % self._logging_iter == 0:
                self._sw.add_scalar("Q-Value", np.mean(q), self._iter)

            # update hidden state in memory according to tau
            if self._new_state_update_tau() > 0.0 and self._replay_memory.size > self._warmup_transitions():
                _, new_next_hidden = self.policy.divide_action_hidden_state(out_new)
                next_env_state = next_state[:, :-self._hidden_state_dim]
                new_next_hidden = new_next_hidden.detach().numpy()
                new_next_state = np.concatenate([next_env_state, new_next_hidden], axis=1)
                new_next_state = self._new_state_update_tau() * new_next_state + \
                                 (1 - self._new_state_update_tau()) * next_state
                self._replay_memory.set_next_state(new_next_state, indices)

            if self._iter % self._logging_iter == 0 and self._sw:
                mean_action = np.mean(action)
                var_action = np.var(action)
                mean_hidden_state = np.mean(next_hidden_state)
                mean_abs_hidden_state = np.mean(np.abs(next_hidden_state))
                var_next_hidden_state = np.var(next_hidden_state)
                abs_diff_hidden_state = np.mean(np.abs(next_hidden_state - hidden_state))
                ones = np.sum(hidden_state == 1.0) / (len(hidden_state[0]) * 100)
                self._sw.add_scalar("Action/Mean", mean_action, self._iter)
                self._sw.add_scalar("Action/Var", var_action, self._iter)
                self._sw.add_scalar("Hidden-State/Mean", mean_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Mean Abs.", mean_abs_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Var", var_next_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Mean Abs. Difference of one hidden state transition",
                                    abs_diff_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Percentage of hidden states that are one.", ones, self._iter)

        self._iter += 1

    def _loss(self, state, prev_action, out_new, log_prob_action, log_prob_hidden):
        next_action, next_hidden_state = self.policy.divide_action_hidden_state(out_new)
        if self._alternating_training_interval_pol_hidden:
            if self._current_turn == "policy":
                q_0 = self._critic_approximator(state[:, self._critic_state_mask], next_action,
                                                next_hidden_state.detach(), prev_action, output_tensor=True, idx=0)
                q_1 = self._critic_approximator(state[:, self._critic_state_mask], next_action,
                                                next_hidden_state.detach(), prev_action, output_tensor=True, idx=1)
            else:
                q_0 = 1e-6 * self._critic_approximator(state[:, self._critic_state_mask], next_action.detach(),
                                                       next_hidden_state, prev_action, output_tensor=True, idx=0)
                q_1 = 1e-6 * self._critic_approximator(state[:, self._critic_state_mask], next_action.detach(),
                                                       next_hidden_state, prev_action, output_tensor=True, idx=1)
        else:
            q_0 = self._critic_approximator(state[:, self._critic_state_mask], next_action, next_hidden_state,
                                            prev_action, output_tensor=True, idx=0)
            q_1 = self._critic_approximator(state[:, self._critic_state_mask], next_action, next_hidden_state,
                                            prev_action, output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return ((self._alpha * log_prob_action + self._alpha_hidden * log_prob_hidden) - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _update_alpha_hidden(self, log_prob):
        alpha_loss = - (self._log_alpha_hidden * (log_prob + self._target_entropy_hidden)).mean()
        self._alpha_hidden_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_hidden_optim.step()

    def _next_q(self, next_state, action, absorbing):
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
        out, log_prob_action, log_prob_next_hidden = \
            self.policy.compute_log_prob_separate(next_state[:, self._policy_state_mask], action)
        next_action, next_hidden_state = self.policy.divide_action_hidden_state(out)

        q = self._target_critic_approximator.predict(
            next_state[:, self._critic_state_mask], next_action, next_hidden_state, action, prediction='min')

        if self._add_hidden_entropy:
            q -= self._alpha_np * log_prob_action + self._alpha_np_hidden * log_prob_next_hidden
        else:
            q -= self._alpha_np * log_prob_action

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

    @property
    def _alpha_hidden(self):
        return self._log_alpha_hidden.exp()

    @property
    def _alpha_np_hidden(self):
        return self._alpha_hidden.detach().cpu().numpy()
