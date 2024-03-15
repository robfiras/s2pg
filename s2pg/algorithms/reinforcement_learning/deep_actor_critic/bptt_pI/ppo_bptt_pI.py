import warnings
import numpy as np

import torch
import torch.nn.functional as F

from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor, update_optimizer_parameters
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import parse_dataset, compute_J, compute_episodes_length
from mushroom_rl.utils.parameters import to_parameter

from s2pg.utils.preprocessors import StandardizationPreprocessor_ext


class PPO_BPTT_pI(Agent):
    """
    Proximal Policy Optimization algorithm.
    Implementation with a recurrent policy and a standard MLP critic. This is supposed to be used
    in the privileged information (pI) setting, where the critic has the full state information, and
    the policy only receives observations.

    Non-Recurrent version first appeared in:
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, hidden_state_dim,
                 n_epochs_policy, batch_size, eps_ppo, lam, dim_env_state, ent_coeff=0.0,
                 policy_state_mask=None, critic_state_mask=None,
                 critic_fit_params=None, standardizer=None, sw=None, truncation_length=5):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            n_epochs_policy ([int, Parameter]): number of policy updates for every dataset;
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            eps_ppo ([float, Parameter]): value for probability ratio clipping;
            lam ([float, Parameter], 1.): lambda coefficient used by generalized
                advantage estimation;
            ent_coeff ([float, Parameter], 1.): coefficient for the entropy regularization term;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict(n_epochs=10) if critic_fit_params is None else critic_fit_params

        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._batch_size = to_parameter(batch_size)
        self._eps_ppo = to_parameter(eps_ppo)

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._lambda = to_parameter(lam)
        self._ent_coeff = to_parameter(ent_coeff)

        self._V = Regressor(TorchApproximator, **critic_params)

        self._standardizer = standardizer

        self._truncation_length = truncation_length
        self._dim_env_state = dim_env_state

        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)
        self._critic_state_mask = critic_state_mask if critic_state_mask is not None else \
            np.concatenate([np.ones(mdp_info.observation_space.shape[0], dtype=bool),
                            np.zeros(hidden_state_dim, dtype=bool)])
        self._hidden_state_dim = hidden_state_dim

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._iter = 1

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom',
            _iter='primitive',
            _dim_env_state='primitive',
            _standardizer='pickle',
            _policy_state_mask='numpy',
            _critic_state_mask='numpy',
            _hidden_state_dim='primitive'
        )

        super().__init__(mdp_info, policy, None)

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        # add the standardization preprocessor
        self._preprocessors.append(StandardizationPreprocessor_ext(mdp_info))

        # add the preprocessor to append the cpg state to the environment state
        self._preprocessors.append(self.append_hidden_state)

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def add_preprocessor(self, preprocessor):
        # for now disable the preprocessor to ensure that appending the cpg state is always done at last
        raise AttributeError("This agent current does not support preprocessors.")

    def add_hidden_state_preprocessor(self):
        if len(self._preprocessors) == 0:   # only the cpg preprocessor is allowed for now, which is why we can check the length
            self._preprocessors.append(self.append_hidden_state)
        else:
            warnings.warn("CPG Preprocessor already included, and will be not added twice.")

    def append_hidden_state(self, x):
        # get latest cpg_state
        cpg_state = self.policy.get_last_hidden_state()
        return np.concatenate([x, cpg_state])

    def fit(self, dataset, **info):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)
        x_seq, u_seq, xn_seq, prev_u_seq, lengths = self.transform_to_sequences(x, u, xn, last, absorbing)

        obs = to_float_tensor(x_seq, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)
        prev_act_seq = to_float_tensor(prev_u_seq)

        # update running mean and std
        if self._standardizer:
            self._standardizer.update_mean_std(x)

        v_target, np_adv = self.compute_gae(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda())
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(obs[:, :, self._policy_state_mask], prev_act_seq, lengths)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x[:, self._critic_state_mask], v_target, **self._critic_fit_params)

        self._update_policy(obs, act, prev_act_seq, lengths, adv, old_log_p)

        # Print fit information
        self._log_info(dataset, x, x_seq, prev_u_seq, lengths, v_target, old_pol_dist)
        self._iter += 1

    def transform_to_sequences(self, states, actions, next_states, last, absorbing):

        s = list()
        a = list()
        ss = list()
        pa = list()         # previous actions
        lengths = list()    # lengths of the sequences

        for i in range(len(states)):

            # determine the begin of a sequence
            begin_seq = np.maximum(i - self._truncation_length + 1, 0)
            end_seq = i + 1

            # maybe the sequence contains more than one trajectory, so we need to cut it so that it contains only one
            lasts_absorbing = np.array(last[begin_seq-1: i], dtype=int) + \
                              np.array(absorbing[begin_seq-1: i], dtype=int)
            begin_traj = np.where(lasts_absorbing > 0)
            sequence_is_shorter_than_requested = len(*begin_traj) > 0
            if sequence_is_shorter_than_requested:
                begin_seq = begin_seq + begin_traj[0][-1]

            # get the sequences
            states_seq = np.array(states[begin_seq:end_seq])
            actions_seq = np.array(actions[begin_seq:end_seq])
            next_states_seq = np.array(next_states[begin_seq:end_seq])
            if sequence_is_shorter_than_requested or begin_seq == 0:
                prev_actions_seq = np.array(actions[begin_seq:end_seq - 1])
                init_prev_action = np.zeros((1, len(actions[0])))
                if len(prev_actions_seq) == 0:
                    prev_actions_seq = init_prev_action
                else:
                    prev_actions_seq = np.concatenate([init_prev_action, prev_actions_seq])
            else:
                prev_actions_seq = np.array(actions[begin_seq - 1:end_seq - 1])

            # apply padding
            length_seq = len(states_seq)
            padded_states = np.concatenate([states_seq, np.zeros((self._truncation_length - states_seq.shape[0],
                                                                  states_seq.shape[1]))])
            padded_next_states = np.concatenate([next_states_seq,
                                                 np.zeros((self._truncation_length - next_states_seq.shape[0],
                                                           next_states_seq.shape[1]))])
            padded_action_seq = np.concatenate([actions_seq,
                                                np.zeros((self._truncation_length - actions_seq.shape[0],
                                                          actions_seq.shape[1]))])
            padded_prev_action = np.concatenate([prev_actions_seq,
                                                 np.zeros((self._truncation_length - prev_actions_seq.shape[0],
                                                           prev_actions_seq.shape[1]))])

            s.append(padded_states)
            a.append(padded_action_seq)
            ss.append(padded_next_states)
            pa.append(padded_prev_action)
            lengths.append(length_seq)

        return np.array(s), np.array(a), np.array(ss), np.array(pa), np.array(lengths)

    def _update_policy(self, obs, act, prev_act_seq, lengths, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, prev_act_seq_i, length_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, prev_act_seq, lengths, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i[:, :, self._policy_state_mask],
                                           act_i, prev_act_seq_i, length_i) - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy_t(obs_i[:, :, self._policy_state_mask])
                loss.backward()
                self._optimizer.step()

    def _log_info(self, dataset, x, x_seq, prev_u, lengths, v_target, old_pol_dist):
        if self._sw:
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            v_pred = torch.tensor(self._V(x[:, self._critic_state_mask]), dtype=torch.float)
            v_err = F.mse_loss(v_pred, torch_v_targets)

            logging_ent = self.policy.entropy(x_seq[:, :, self._policy_state_mask])
            new_pol_dist = self.policy.distribution(x_seq[:, :, self._policy_state_mask], prev_u, lengths)
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(
                new_pol_dist, old_pol_dist))
            avg_rwd = np.mean(compute_J(dataset))
            L = int(np.round(np.mean(compute_episodes_length(dataset))))

            self._sw.add_scalar('Training Details/Training Reward', avg_rwd, self._iter)
            self._sw.add_scalar('Training Details/Mean Episode Length', L, self._iter)
            self._sw.add_scalar('Training Details/Value-Function Loss', v_err, self._iter)
            self._sw.add_scalar('Training Details/Mean Value', torch.mean(v_pred), self._iter)
            self._sw.add_scalar('Training Details/Entropy', logging_ent, self._iter)
            self._sw.add_scalar('Training Details/KL', logging_kl, self._iter)

    def _post_load(self):
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))

    def compute_gae(self, V, s, ss, r, absorbing, last, gamma, lam):
        """
        Function to compute Generalized Advantage Estimation (GAE)
        and new value function target over a dataset.

        "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation".
        Schulman J. et al.. 2016.

        Args:
            V (Regressor): the current value function regressor;
            s (numpy.ndarray): the set of states in which we want
                to evaluate the advantage;
            ss (numpy.ndarray): the set of next states in which we want
                to evaluate the advantage;
            r (numpy.ndarray): the reward obtained in each transition
                from state s to state ss;
            absorbing (numpy.ndarray): an array of boolean flags indicating
                if the reached state is absorbing;
            last (numpy.ndarray): an array of boolean flags indicating
                if the reached state is the last of the trajectory;
            gamma (float): the discount factor of the considered problem;
            lam (float): the value for the lamba coefficient used by GEA
                algorithm.
        Returns:
            The new estimate for the value function of the next state
            and the estimated generalized advantage.
        """
        v = V(s[:, self._critic_state_mask])
        v_next = V(ss[:, self._critic_state_mask])
        gen_adv = np.empty_like(v)
        for rev_k in range(len(v)):
            k = len(v) - rev_k - 1
            if last[k] or rev_k == 0:
                gen_adv[k] = r[k] - v[k]
                if not absorbing[k]:
                    gen_adv[k] += gamma * v_next[k]
            else:
                gen_adv[k] = r[k] + gamma * v_next[k] - v[k] + gamma * lam * gen_adv[k + 1]
        return gen_adv + v, gen_adv
