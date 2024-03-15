import warnings
import numpy as np

from ..policy_gradient import BatchedStatefulPolicyGradient
from mushroom_rl.utils.dataset import compute_J, parse_dataset


class BatchedREINFORCE_BPTT(BatchedStatefulPolicyGradient):
    """
    REINFORCE algorithm.
    "Simple Statistical Gradient-Following Algorithms for Connectionist
    Reinforcement Learning", Williams R. J.. 1992.
    """
    def __init__(self, mdp_info, policy, optimizer, truncation_length, features=None, sw=None, use_env_state=True, logger=None, int_log=1):
        super().__init__(mdp_info, policy, optimizer, features)
        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.baseline_num = list()
        self.baseline_den = list()
        self._truncation_length = truncation_length
        self._use_env_state = use_env_state
        self._logger = logger
        self._int_log = int_log

        # add the preprocessor to append the cpg state to the environment state
        self._preprocessors.append(self.append_hidden_state)

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        self._iter = 0

        self._add_save_attr(
            sum_d_log_pi='numpy',
            list_sum_d_log_pi='pickle',
            baseline_num='pickle',
            baseline_den='pickle',
            _iter='primitive',
            _truncation_length='primitive'
        )

    def add_preprocessor(self, preprocessor):
        # for now disable the preprocessor to ensure that appending the hidden state is always done at last
        raise AttributeError("This agent current does not support preprocessors.")

    def add_hidden_state_preprocessor(self):
        if len(self._preprocessors) == 0:   # only the hidden preprocessor is allowed for now, which is why we can check the length
            self._preprocessors.append(self.append_hidden_state)
        else:
            warnings.warn("Hidden Preprocessor already included, and will be not added twice.")

    def append_hidden_state(self, x):
        # get latest cpg_state
        hidden_state = self.policy.get_last_hidden_state()
        return np.concatenate([x, hidden_state])

    def _compute_gradient(self, J):
        baseline = np.mean(self.baseline_num, axis=0) / np.mean(self.baseline_den, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.
        grad_J_episode = list()
        for i, J_episode in enumerate(J):
            sum_d_log_pi = self.list_sum_d_log_pi[i]
            grad_J_episode.append(sum_d_log_pi * (J_episode - baseline))

        grad_J = np.mean(grad_J_episode, axis=0)
        self.list_sum_d_log_pi = list()
        self.baseline_den = list()
        self.baseline_num = list()

        # do some logging
        entropy = self.policy.entropy()

        if self._sw and self._iter % self._int_log == 0:
            var = np.mean(np.var(grad_J_episode, axis=0))
            self._sw.add_scalar("Policy/Mean Gradient", np.mean(grad_J_episode), self._iter)
            self._sw.add_scalar("Policy/Var Gradient", var, self._iter)
            if self._logger is not None:
                for logger in self._logger:
                    logger.log_numpy(Var_Gradient=var)
            self._sw.add_scalar("Entropy/Entropy", entropy, self._iter)
            self._sw.add_scalar("Entropy/Mean std ", np.mean(np.exp(self.policy._logstd)),
                                self._iter)
        self._iter += 1

        return grad_J

    def _all_step_updates(self, dataset):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        x_seq, u_seq, xn_seq, prev_u_seq, lengths = self.transform_to_sequences(x, u, xn, last, absorbing)

        if not self._use_env_state:
            # get env and cpg states
            _, x = self.policy.divide_state_to_env_hidden_batch(x)
            _, xn = self.policy.divide_state_to_env_hidden_batch(xn)

        # calculate the sum for each episode
        split_idx = np.squeeze(np.argwhere(last == 1))[:-1]+1

        # divide into episodes
        x_episodes = np.split(x_seq, split_idx)
        xn_episodes = np.split(xn_seq, split_idx)
        u_episodes = np.split(u, split_idx)
        prev_u_episodes = np.split(prev_u_seq, split_idx)
        lengths_episodes = np.split(lengths, split_idx)

        # calculate the J's
        Js = compute_J(dataset, gamma=self.mdp_info.gamma)

        for x_e, xn_e, u_e, prev_u_e, l_e, J in zip(x_episodes, xn_episodes, u_episodes,
                                                    prev_u_episodes, lengths_episodes, Js):
            # calculate the step-based gradients
            sum_d_log_pi = self.policy.diff_log_batch(x_e, u_e, xn_e, prev_u_e, l_e)
            self.list_sum_d_log_pi.append(sum_d_log_pi)
            squared_sum_d_log_pi = np.square(sum_d_log_pi)
            self.baseline_num.append(squared_sum_d_log_pi * J)
            self.baseline_den.append(squared_sum_d_log_pi)

        return Js

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
            padded_next_states = np.concatenate([next_states_seq, np.zeros((self._truncation_length - next_states_seq.shape[0],
                                                              next_states_seq.shape[1]))])
            padded_action_seq = np.concatenate([actions_seq, np.zeros((self._truncation_length - actions_seq.shape[0],
                                                              actions_seq.shape[1]))])
            padded_prev_action = np.concatenate([prev_actions_seq, np.zeros((self._truncation_length - prev_actions_seq.shape[0],
                                                              prev_actions_seq.shape[1]))])

            s.append(padded_states)
            a.append(padded_action_seq)
            ss.append(padded_next_states)
            pa.append(padded_prev_action)
            lengths.append(length_seq)

        return np.array(s), np.array(a), np.array(ss), np.array(pa), np.array(lengths)

    @staticmethod
    def get_prev_action(actions, lasts, absorbings):
        pa = list()
        for i in range(len(actions)):
            if i == 0 or lasts[i - 1] or absorbings[i - 1]:
                pa.append(np.zeros(len(actions[0])))
            else:
                pa.append(actions[i - 1])

        return np.array(pa)
