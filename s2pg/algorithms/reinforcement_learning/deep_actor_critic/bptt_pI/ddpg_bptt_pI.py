import warnings
import numpy as np
import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import to_float_tensor
from s2pg.utils.replay_memory import SequenceReplayMemory

from copy import deepcopy


class DDPG_BPTT_pI(DeepAC):
    """
    Deep Deterministic Policy Gradient algorithm.
    Implementation with a recurrent policy and a standard MLP critic. This is supposed to be used
    in the privileged information (pI) setting, where the critic has the full state information, and
    the policy only receives observations.

    Non-Recurrent version first appeared in:
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params, actor_optimizer, truncation_length,
                 critic_params, batch_size, initial_replay_size, max_replay_size, tau, hidden_state_dim, policy_delay=1,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None,
                 sw=None, logging_iter=1000, max_grad_norm_clip=10.0, warmup_transitions=0,
                 policy_state_mask=None, critic_state_mask=None,):
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
        self._policy_delay = to_parameter(policy_delay)
        self._fit_count = 0
        self._warmup_transitions = to_parameter(warmup_transitions)

        self._truncation_length = truncation_length
        self._replay_memory = SequenceReplayMemory(truncation_length, initial_replay_size, max_replay_size)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)
        self._critic_state_mask = critic_state_mask if critic_state_mask is not None else\
            np.concatenate([np.ones(mdp_info.observation_space.shape[0], dtype=bool),
                            np.zeros(hidden_state_dim, dtype=bool)])

        # --- create online and target actors ---
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
            _policy_state_mask='numpy',
            _critic_state_mask='numpy'
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
            state, action, reward, next_state, absorbing, _, prev_action, lengths, action_seq =\
                self._replay_memory.get(self._batch_size())

            q_next = self._next_q(next_state, absorbing, action_seq, lengths)
            q = reward + self.mdp_info.gamma * q_next

            rel_indices = lengths.reshape(-1, 1, 1) - 1
            last_state = np.squeeze(np.take_along_axis(state, rel_indices, axis=1), axis=1)

            self._critic_approximator.fit(last_state[:, self._critic_state_mask], action, q,
                                          **self._critic_fit_params)

            if self._fit_count % self._policy_delay() == 0 and self._replay_memory.size > self._warmup_transitions():
                loss = self._loss(state, prev_action, lengths)
                self._optimize_actor_parameters(loss)
                if self._iter % self._logging_iter == 0 and self._sw:
                    grads = []
                    for param in self._actor_approximator.model.network.parameters():
                       grads.append(param.grad.view(-1))
                    grads = torch.cat(grads)
                    norm = grads.norm(dim=0, p=2)
                    self._sw.add_scalar('Gradients/Norm2 Gradient Q wrt. Pi-parameters', norm,
                                       self._iter)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
            self._update_target(self._actor_approximator,
                                self._target_actor_approximator)

            self._fit_count += 1
            self._iter += 1

            hidden_states = state[:, -1, self.mdp_info.observation_space.shape[0]:]

            if self._iter % self._logging_iter == 0 and self._sw:
                mean_q = np.mean(q_next)
                mean_action = np.mean(action)
                var_action = np.var(action)
                mean_hidden_state = np.mean(hidden_states)
                mean_abs_hidden_state = np.mean(np.abs(hidden_states))
                var_next_hidden_state = np.var(hidden_states)
                self._sw.add_scalar("Mean Q", mean_q, self._iter)
                self._sw.add_scalar("Action/Mean", mean_action, self._iter)
                self._sw.add_scalar("Action/Var", var_action, self._iter)
                self._sw.add_scalar("Hidden-State/Mean", mean_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Mean Abs.", mean_abs_hidden_state, self._iter)
                self._sw.add_scalar("Hidden-State/Var", var_next_hidden_state, self._iter)

    def _loss(self, state, prev_action, lengths):
        action, _ = self._actor_approximator.model.network(to_float_tensor(state),
                                                           to_float_tensor(prev_action),
                                                           torch.tensor(lengths))
        q = self._critic_approximator(state, action, to_float_tensor(prev_action),
                                      torch.tensor(lengths), output_tensor=True,
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
        # todo: implement prev_action and length for ddpg as well.
        action, next_next_hidden_state = self._target_actor_approximator.model.network(to_float_tensor(next_state))
        action = action.detach().cpu().numpy()
        next_next_hidden_state = next_next_hidden_state.detach().cpu().numpy()
        q = self._target_critic_approximator.predict(next_state, action, next_next_hidden_state,
                                                     **self._critic_predict_params)
        q *= 1 - absorbing

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
