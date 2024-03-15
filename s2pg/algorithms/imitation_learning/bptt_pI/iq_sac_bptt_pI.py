from copy import deepcopy
from tqdm import tqdm
import warnings

import numpy as np
import torch
import torch.optim as optim
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from s2pg.utils.replay_memory import SequenceReplayMemory
from s2pg.policy.recurrent_policy import SACBPTTPolicy

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC


class IQ_SAC_BPTT_pI(DeepAC):

    """
    Implementation of IQ-Learn based on SAC.
    Implementation with a recurrent policy and a standard MLP critic. This is supposed to be used
    in the privileged information (pI) setting, where the critic has the full state information, and
    the policy only receives observations.

    "IQ-Learn: Inverse soft-Q Learning for Imitation"
    Divyansh Garg, Shuvam Chakraborty,  Chris Cundy, Jiaming Song and Stefano Ermon (2021)

    The implementation of the critic update is based on the repository of the paper authors:
    https://github.com/Div99/IQ-Learn/blob/main/iq_learn
    """

    def __init__(self, mdp_info, actor_params, hidden_state_dim, expert_dataset,
                 actor_optimizer, critic_params, batch_size, sw, use_target,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, delay_pi=1, delay_Q=1,
                 reg_mult=(1 / (4 * 0.5)), log_std_min=-20, log_std_max=2, target_entropy=None, ext_normalizer=None,
                 critic_fit_params=None, learnable_alpha=False, init_alpha=0.001, critic_state_mask=None,
                 policy_state_mask=None, plcy_loss_mode="value",
                 regularizer_mode="exp_and_plcy", logging_iter=1, truncation_length=5,
                 n_fits=1,  train_policy_only_on_own_states=False, use_cuda=False,
                 treat_absorbing_states=False, policy_type_params=None, **kwargs):

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape[0]).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._hidden_state_dim = hidden_state_dim

        self._replay_memory = SequenceReplayMemory(truncation_length=truncation_length,
                                                   initial_replay_size=initial_replay_size,
                                                   max_replay_size=max_replay_size)
        self._replay_memory_expert = SequenceReplayMemory(truncation_length=truncation_length,
                                                          initial_replay_size=0,
                                                          max_replay_size=len(expert_dataset["states"]))

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] <= 1 # here it differs from sac, as we only take one critic

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        actor_approximator = Regressor(TorchApproximator,
                                       **actor_params)
        self._iter = 1

        policy_type_params = dict() if policy_type_params is None else policy_type_params
        policy = SACBPTTPolicy(approximator=actor_approximator,
                               action_dim=mdp_info.action_space.shape[0],
                               hidden_state_dim=hidden_state_dim,
                               lows=mdp_info.action_space.low,
                               highs=mdp_info.action_space.high,
                               log_std_min=log_std_min,
                               log_std_max=log_std_max,
                               **policy_type_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        policy_parameters = actor_approximator.model.network.parameters()

        ### End of SAC constructor, start of IQ_Learn's

        # define the optimizer
        net_params = self._critic_approximator.model.network.parameters()
        self._critic_optimizer = critic_params["optimizer"]["class"](net_params, **critic_params["optimizer"]["params"])

        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)
        self._critic_state_mask = critic_state_mask if critic_state_mask is not None else\
            np.concatenate([np.ones(mdp_info.observation_space.shape[0], dtype=bool),
                            np.zeros(hidden_state_dim, dtype=bool)])

        # use target for critic update
        self._use_target = use_target

        # check if alpha should be learnable or not
        self._learnable_alpha = learnable_alpha
        self._log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        self._plcy_loss_mode = plcy_loss_mode
        self._regularizer_mode = regularizer_mode
        self._reg_mult = reg_mult
        self._use_cuda = use_cuda
        self._delay_pi = delay_pi
        self._delay_Q = delay_Q
        self._train_policy_only_on_own_states = train_policy_only_on_own_states
        self._n_fits = n_fits
        self._treat_absorbing_states = treat_absorbing_states

        self._logging_iter = logging_iter

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        self._epoch_counter = 1

        self.ext_normalizer = ext_normalizer
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
            _reg_mult='primitive',
            ext_normalizer='pickle',
            _hidden_state_dim='primitive',
            _truncation_length='primitive'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        # add the preprocessor to append the hidden state to the environment state
        self._preprocessors.append(self.append_hidden_state)

        # initialuze expert replay memory
        self._init_expert_replay_memory(expert_dataset)

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

    def _init_expert_replay_memory(self, demostrations):

        states = demostrations["states"]
        actions = demostrations["actions"]
        rewards = demostrations["rewards"]
        next_states = demostrations["next_states"]
        absorbings = demostrations["absorbing"]
        lasts = demostrations["last"]
        last_hidden_state = np.zeros(self._hidden_state_dim)
        last_action = np.zeros(self.mdp_info.action_space.shape[0])

        print("\nPredicting hidden state for expert trajectories ...")
        for s, a, r, ss, ab, last in tqdm(zip(states, actions, rewards, next_states,
                                              absorbings, lasts), total=len(states)):

            # predict new hidden state and append to next_state
            ext_s = np.concatenate([s, last_hidden_state])
            _, new_next_hidden = \
                self.policy.compute_action_hidden_and_log_prob(np.atleast_2d(ext_s[self._policy_state_mask]),
                                                               np.atleast_2d(last_action), compute_log_prob=False,
                                                               return_hidden_state=True)
            new_next_hidden = np.squeeze(new_next_hidden)
            ext_ss = np.concatenate([ss, new_next_hidden])

            # construct a sample to append to the replay memory
            sample = [ext_s, a, r, ext_ss, ab, last]
            self._replay_memory_expert.add([sample])

            if last:
                last_hidden_state = np.zeros(self._hidden_state_dim)
                last_action = np.zeros(self.mdp_info.action_space.shape[0])
            else:
                last_hidden_state = new_next_hidden
                last_action = a

    def fit(self, dataset, **info):

        # add to replay memory
        self._replay_memory.add(dataset)

        if self._replay_memory.initialized:

            for i in range(self._n_fits):

                # sample batch from policy replay buffer
                state, action, reward, next_state, absorbing, _, prev_a, lengths, action_seq = \
                    self._replay_memory.get(self._batch_size())

                # get expert data
                demo_obs, demo_act, demo_reward, demo_nobs, demo_absorbing, _, demo_prev_a,\
                    demo_lengths, demo_action_seq = self._replay_memory_expert.get(self._batch_size())

                # prepare data for IQ update
                input_states = to_float_tensor(np.concatenate([state, demo_obs.astype(np.float32)]))
                input_actions = to_float_tensor(np.concatenate([action, demo_act.astype(np.float32)]))
                input_n_states = to_float_tensor(np.concatenate([next_state,
                                                                 demo_nobs.astype(np.float32)]))
                input_absorbing = to_float_tensor(np.concatenate([absorbing, demo_absorbing.astype(np.float32)]))
                input_prev_action = to_float_tensor(np.concatenate([prev_a, demo_prev_a.astype(np.float32)]))
                input_action_seq = to_float_tensor(np.concatenate([action_seq, demo_action_seq.astype(np.float32)]))
                input_lengths = torch.tensor(np.concatenate([lengths, demo_lengths]))
                is_expert = torch.concat([torch.zeros(len(state), dtype=torch.bool),
                                          torch.ones(len(state), dtype=torch.bool)])

                # make IQ update
                self.iq_update(input_states, input_actions, input_n_states, input_absorbing,
                               input_prev_action, input_action_seq, input_lengths, is_expert)

        self._iter += 1

    def iq_update(self, input_states, input_actions, input_n_states, input_absorbing, input_prev_action,
                  input_action_seq, input_lengths, is_expert):

        # update Q function
        if self._iter % self._delay_Q == 0:
            self.update_Q_function(input_states, input_actions, input_n_states, input_absorbing, input_action_seq,
                                   input_prev_action, input_lengths, is_expert)

        # update policy
        if self._replay_memory.size > self._warmup_transitions() and self._iter % self._delay_pi == 0:
            self.update_policy(input_states, input_prev_action, input_lengths, is_expert)

        if self._iter % self._delay_Q == 0:
            self._update_all_targets()

    def update_Q_function(self, input_states, input_actions, input_n_states, input_absorbing, input_prev_action,
                          input_action_seq, input_lengths, is_expert):

        loss1, loss2, chi2_loss = self._lossQ(input_states, input_actions, input_n_states, input_absorbing,
                                              input_prev_action, input_action_seq, input_lengths, is_expert)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('IQ-Loss/Loss1', loss1, self._iter)
            self.sw_add_scalar('IQ-Loss/Loss2', loss2, self._iter)
            self.sw_add_scalar('IQ-Loss/Chi2 Loss', chi2_loss, self._iter)
            self.sw_add_scalar('IQ-Loss/Alpha', self._alpha, self._iter)

    def update_policy(self, input_states, input_prev_action, lengths, is_expert):

        action_new, log_prob = self.policy.compute_action_hidden_and_log_prob_t(input_states[:, :, self._policy_state_mask],
                                                                            input_prev_action, lengths)
        rel_indices = lengths.view(-1, 1, 1) - 1
        last_states = torch.squeeze(torch.take_along_dim(input_states, rel_indices, dim=1), dim=1)
        loss = self._actor_loss(last_states, action_new, log_prob)
        self._optimize_actor_parameters(loss)

        if self._iter % self._logging_iter == 0:
            grads = []
            for param in self.policy._approximator.model.network.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            norm = grads.norm(dim=0, p=2)
            self.sw_add_scalar('Gradients/Norm2 Gradient Q wrt. Pi-parameters', norm,
                               self._iter)
            self.sw_add_scalar('Actor/Loss', loss, self._iter)
            self.sw_add_scalar('Actor/Entropy Expert States Action', torch.mean(-log_prob[is_expert]).detach().item(),
                               self._iter)
            self.sw_add_scalar('Actor/Entropy Policy States Action', torch.mean(-log_prob[~is_expert]).detach().item(),
                               self._iter)

        if self._learnable_alpha:
            self._update_alpha(log_prob[~is_expert].detach())

    def _lossQ(self, obs, act, next_obs, absorbing, prev_action, action_seq, lengths, is_expert):

        rel_indices = lengths.view(-1, 1, 1) - 1
        last_obs = torch.squeeze(torch.take_along_dim(obs, rel_indices, dim=1), dim=1)
        last_next_obs = torch.squeeze(torch.take_along_dim(next_obs, rel_indices, dim=1), dim=1)

        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(last_obs[:, self._critic_state_mask], act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs, last_next_obs, action_seq, lengths)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs, last_next_obs, action_seq, lengths).detach()

        y = (1 - torch.unsqueeze(absorbing, 1)) * gamma.detach() * next_v

        reward = (current_Q - y)
        exp_reward = reward[is_expert]
        loss_term1 = -exp_reward.mean()

        # do the logging
        self.logging_loss(current_Q, y, reward, is_expert, obs, act, absorbing)

        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        if self._plcy_loss_mode == "value":
            V = self.getV(obs, last_obs, prev_action, lengths)
            value = (V - y)
            self.sw_add_scalar('V for policy on all states', V.mean(), self._iter)
            value_loss = value
            loss_term2 = value_loss.mean()
        elif self._plcy_loss_mode == "value_expert":
            V = self.getV(obs, last_obs, prev_action, lengths)
            value = (V - y)
            self.sw_add_scalar('V for policy on all states', V.mean(), self._iter)
            value_loss = value
            value_loss_exp = value_loss[is_expert]
            loss_term2 = value_loss_exp.mean()
        elif self._plcy_loss_mode == "value_policy":
            V = self.getV(obs, last_obs, prev_action, lengths)
            value = (V - y)
            self.sw_add_scalar('V for policy on all states', V.mean(), self._iter)
            value_loss = value
            value_loss_plcy = value_loss[~is_expert]
            loss_term2 = value_loss_plcy.mean()
        elif self._plcy_loss_mode == "q_old_policy":
            reward = (current_Q - y)
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean()
        elif self._plcy_loss_mode == "v0":
            V = self.getV(obs, last_obs, prev_action, lengths)
            value = (V - y)
            self.sw_add_scalar('V for policy on all states', V.mean(), self._iter)
            value_loss_v0 = (1-gamma.detach()) * V[is_expert]
            loss_term2 = value_loss_v0.mean()
        else:
            raise ValueError("Undefined policy loss mode: %s" % self._plcy_loss_mode)

        # regularize
        absorbing = torch.unsqueeze(absorbing, 1)
        chi2_loss = self.regularizer_loss(absorbing, reward, gamma, is_expert,
                                          treat_absorbing_states=self._treat_absorbing_states)

        loss_Q = loss_term1 + loss_term2 + chi2_loss
        self.update_Q_parameters(loss_Q)

        if self._iter % self._logging_iter == 0:
            grads = []
            for param in self._critic_approximator.model.network.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            norm = grads.norm(dim=0, p=2)
            self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_term1, loss_term2, chi2_loss

    def regularizer_loss(self, absorbing, reward, gamma, is_expert, treat_absorbing_states=False):
        # choose whether to treat absorbing states or not
        if treat_absorbing_states:
            reg_absorbing = absorbing
        else:
            reg_absorbing = torch.zeros_like(absorbing)

        if self._regularizer_mode == "exp_and_plcy":
            chi2_loss = ((1 - reg_absorbing) * torch.tensor(self._reg_mult) * torch.square(reward)
                         + reg_absorbing * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward))).mean()
        elif self._regularizer_mode == "exp":
            chi2_loss = ((1 - reg_absorbing[is_expert]) * torch.tensor(self._reg_mult) * torch.square(reward[is_expert])
                         + reg_absorbing[is_expert] * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward[is_expert]))).mean()
        elif self._regularizer_mode == "plcy":
            chi2_loss = ((1 - reg_absorbing[~is_expert]) * torch.tensor(self._reg_mult) * torch.square(reward[~is_expert])
                         + reg_absorbing[~is_expert] * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward[~is_expert]))).mean()
        elif self._regularizer_mode == "off":
            chi2_loss = 0.0
        else:
            raise ValueError("Undefined regularizer mode %s." % (self._regularizer_mode))

        return chi2_loss

    def update_Q_parameters(self, loss):
        loss = loss.mean()
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()

    def getV(self, obs, last_obs, prev_a, lengths):
        with torch.no_grad():
            action, log_prob = self.policy.compute_action_hidden_and_log_prob_t(obs[:, :, self._policy_state_mask],
                                                                                prev_a, lengths)

            log_prob = torch.unsqueeze(log_prob, 1)
        current_Q = self._critic_approximator(last_obs[:, self._critic_state_mask], action.detach(),
                                              output_tensor=True)

        current_V = current_Q - self._alpha.detach() * log_prob.detach()

        return current_V

    def get_targetV(self, obs, last_obs, prev_a, lengths):
        with torch.no_grad():
            action, log_prob = self.policy.compute_action_hidden_and_log_prob_t(obs[:, :, self._policy_state_mask],
                                                                                prev_a, lengths)
            log_prob = torch.unsqueeze(log_prob, 1)

        target_Q = self._target_critic_approximator(last_obs[:,  self._critic_state_mask], action.detach(),
                                                    output_tensor=True)

        target_V = target_Q - self._alpha.detach() * log_prob.detach()

        return target_V

    def _actor_loss(self, last_state, action_new, log_prob_action):
        log_prob_action = torch.unsqueeze(log_prob_action, 1)
        q = self._critic_approximator(last_state[:, self._critic_state_mask], action_new, output_tensor=True)
        soft_q = (self._alpha * log_prob_action) - q
        return soft_q.mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()

    def logging_loss(self, current_Q, y, reward, is_expert, obs, act, absorbing):

        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Action-Value/Q for expert', current_Q[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for expert', torch.square(current_Q[is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q for policy', current_Q[~is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for policy', torch.square(current_Q[~is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward', reward.mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Expert', reward[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Policy', reward[~is_expert].mean(), self._iter)

            Q_exp = current_Q[is_expert]
            Q_plcy = current_Q[~is_expert]
            abs_exp = absorbing[is_expert].bool()
            abs_plcy = absorbing[~is_expert].bool()
            self.sw_add_scalar('Action-Value/Q Absorbing state exp', torch.mean(Q_exp[abs_exp]), self._iter)
            self.sw_add_scalar('Action-Value/Q Absorbing state plcy', torch.mean(Q_plcy[abs_plcy]), self._iter)

            # norm
            w = self._critic_approximator.get_weights()
            self.sw_add_scalar("Action-Value/Norm of Q net: ",np.linalg.norm(w), self._iter)
            self.sw_add_scalar('Targets/expert data', y[is_expert].mean(), self._iter)
            self.sw_add_scalar('Targets/policy data', y[~is_expert].mean(), self._iter)

            # log mean of each action
            n_actions = len(act[0])
            for i in range(n_actions):
                self.sw_add_scalar('All Actions means/action %d expert' % i, act[is_expert, i].mean(),
                                   self._iter)
                self.sw_add_scalar('All Actions means/action %d policy' % i, act[~is_expert, i].mean(),
                                   self._iter)
                self.sw_add_scalar('All Actions variances/action %d expert' % i, torch.var(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions variances/action %d policy' % i, torch.var(act[~is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d expert' % i, torch.min(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d policy' % i, torch.min(act[~is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d expert' % i, torch.min(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d policy' % i, torch.min(act[~is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions maxs/action %d expert' % i, torch.max(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions maxs/action %d policy' % i, torch.max(act[~is_expert, i]),
                                   self._iter)

    def _update_all_targets(self):
        self._update_target(self._critic_approximator,
                            self._target_critic_approximator)

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    def sw_add_scalar(self, name, val, iter):
        if self._iter % self._logging_iter == 0:
            self._sw.add_scalar(name, val, self._iter)
