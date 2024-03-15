import warnings
import numpy as np
import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from mushroom_rl.utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import to_float_tensor
from s2pg.utils.replay_memory import ReplayMemoryPrevAction_with_returnv2

from copy import deepcopy


class DDPG_RS(DeepAC):
    """
    Deep Deterministic Policy Gradient algorithm formulated with the stochastic stateful policy gradient.
    Implementation with a recurrent stochastic policy (hence RS) and a critic that takes the policy states
    into account. This can be used in settings with and without privileged information, even though without
    the privileged information the performance is poor (see paper).

    Non-Recurrent version first appeared in:
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params, actor_optimizer, hidden_state_dim,
                 critic_params, batch_size, initial_replay_size, max_replay_size, tau, tau_actor, policy_delay=1,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None, Q_clipping=False,
                 sw=None, logging_iter=1000, max_grad_norm_clip=10.0, warmup_transitions=0,
                 new_state_update_tau=0.0, policy_state_mask=None, critic_state_mask=None,):
        """
        Constructor.
        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ((float, Parameter)): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 1): the number of updates of the critic after
                which an actor update is implemented;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
            actor_predict_params (dict, None): parameters for the prediction with the
                actor approximator;
            critic_predict_params (dict, None): parameters for the prediction with the
                critic approximator.
        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._tau_actor = to_parameter(tau_actor)
        self._policy_delay = to_parameter(policy_delay)
        self._new_state_update_tau = to_parameter(new_state_update_tau)
        self._hidden_state_dim = hidden_state_dim
        self._fit_count = 0
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._R_min = 0.0
        self._R_max = 0.0
        self._Q_clipping = Q_clipping

        self._replay_memory = ReplayMemoryPrevAction_with_returnv2(initial_size=initial_replay_size,
                                                                   max_size=max_replay_size)

        #self._replay_memory = ReplayMemoryPrevAction_with_return(mdp_info=mdp_info, initial_size=initial_replay_size,
        #                                                         max_size=max_replay_size)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        #--- create online and target actors ---
        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)
        self._init_target(self._actor_approximator,
                          self._target_actor_approximator)

        policy = policy_class(mdp_info=mdp_info,
                              mu_approximator=self._actor_approximator,
                              **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._max_grad_norm_clip = max_grad_norm_clip

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
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_actor_approximator='mushroom',
            _new_state_update_tau='mushroom',
            _policy_state_mask='numpy',
            _critic_state_mask='numpy',
            _hidden_state_dim='primitive'
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
        if len(self._preprocessors) == 0:   # only the hidden state preprocessor is allowed for now, which is why we can check the length
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

            _, hidden_state = self.policy.divide_state_to_env_hidden_batch(state[:, self._policy_state_mask])
            _, next_hidden_state = self.policy.divide_state_to_env_hidden_batch(next_state[:, self._policy_state_mask])

            q_next = self._next_q(next_state, absorbing, action)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state[:, self._critic_state_mask], action, next_hidden_state, prev_action, q,
                                          **self._critic_fit_params)

            if self._fit_count % self._policy_delay() == 0 and self._replay_memory.size > self._warmup_transitions():
                loss, new_next_hidden_state = self._loss(state, prev_action)
                self._optimize_actor_parameters(loss)
                if self._iter % self._logging_iter == 0 and self._sw:
                    grads = []
                    for param in self._actor_approximator.model.network.parameters():
                        if param.grad is not None:
                            grads.append(param.grad.view(-1))
                    grads = torch.cat(grads)
                    mean_abs = torch.mean(torch.abs(grads))
                    max = torch.max(torch.abs(grads))
                    norm = grads.norm(dim=0, p=2)
                    self._sw.add_scalar('Gradients/Norm2 Gradient Q wrt. Pi-parameters', norm,
                                        self._iter)
                    self._sw.add_scalar('Gradients/Mean-Abs Gradient Q wrt. Pi-parameters', mean_abs,
                                        self._iter)
                    self._sw.add_scalar('Gradients/Max Gradient Q wrt. Pi-parameters', max,
                                        self._iter)

            # update hidden state in memory according to tau
            if self._fit_count % self._policy_delay() == 0 and self._new_state_update_tau() > 0.0\
                    and self._replay_memory.size > self._warmup_transitions():
                next_env_state = next_state[:, :-self._hidden_state_dim]
                new_next_hidden = new_next_hidden_state.detach().numpy()
                new_next_state = np.concatenate([next_env_state, new_next_hidden], axis=1)
                new_next_state = self._new_state_update_tau() * new_next_state + \
                                 (1 - self._new_state_update_tau()) * next_state
                self._replay_memory.set_next_state(new_next_state, indices)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
            self._update_target_actor(self._actor_approximator,
                                      self._target_actor_approximator)

            self._fit_count += 1
            self._iter += 1

            if self._iter % self._logging_iter == 0 and self._sw:
                mean_q = np.mean(q_next)
                mean_action = np.mean(action)
                var_action = np.var(action)
                mean_hidden_state = np.mean(next_hidden_state)
                mean_abs_hidden_state = np.mean(np.abs(next_hidden_state))
                var_next_hidden_state = np.var(next_hidden_state)
                abs_diff_hidden_state = np.mean(np.abs(next_hidden_state - hidden_state))
                ones = np.sum(hidden_state == 1.0) / (len(hidden_state[0]) * 100)
                self._sw.add_scalar("Mean Q", mean_q, self._iter)
                self._sw.add_scalar("Action/Mean", mean_action, self._iter)
                self._sw.add_scalar("Action/Var", var_action, self._iter)
                self._sw.add_scalar("Hidden-State/Mean", mean_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Mean Abs.", mean_abs_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Var", var_next_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Mean Abs. Difference of one hidden state transition",
                                    abs_diff_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Percentage of hidden states that are one.", ones, self._iter)
                self._sw.add_scalar("Reward Bounds/R_max ", self._R_max, self._iter)
                self._sw.add_scalar("Reward Bounds/R_min ", self._R_min, self._iter)

    def _update_target_actor(self, online, target):
        for i in range(len(target)):
            weights = self._tau_actor() * online[i].get_weights()
            weights += (1 - self._tau_actor.get_value()) * target[i].get_weights()
            target[i].set_weights(weights)

    def _loss(self, state):
        action, next_hidden_state = self._actor_approximator.model.network(to_float_tensor(state[:, self._policy_state_mask]))
        q = self._critic_approximator(state[:, self._critic_state_mask], action, next_hidden_state, output_tensor=True,
                                      **self._critic_predict_params)
        return -q.mean(), next_hidden_state

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
        action, next_next_hidden_state = self._target_actor_approximator.model.network(to_float_tensor(next_state[:, self._policy_state_mask]))
        action = action.detach().cpu().numpy()
        next_next_hidden_state = next_next_hidden_state.detach().cpu().numpy()
        q = self._target_critic_approximator.predict(next_state[:, self._critic_state_mask], action,
                                                     next_next_hidden_state, **self._critic_predict_params)
        q *= 1 - absorbing

        if self._Q_clipping:
            q = np.clip(q, self._R_min/(1-self.mdp_info.gamma), self._R_max/(1-self.mdp_info.gamma))

        return q

    def _optimize_actor_parameters(self, loss):
        """
        Method used to update actor parameters to maximize a given loss.
        Args:
            loss (torch.tensor): the loss computed by the algorithm.
        """
        self._optimizer.zero_grad()
        loss.backward()
        self._clip_gradient()
        self._optimizer.step()

    def _post_load(self):
        self._actor_approximator = self.policy._mu_approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())

    @property
    def started_training(self):
        return self._replay_memory.initialized