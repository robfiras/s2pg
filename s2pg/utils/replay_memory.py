import numpy as np
from copy import deepcopy
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.dataset import compute_J, arrays_as_dataset
from mushroom_rl.core import Serializable


class BaseReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self.buffer_created = False

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive!',
            _full='primitive!',
            _states='pickle!',
            _actions='pickle!',
            _rewards='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
            _last='pickle!'
        )

    def add(self, dataset, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert n_steps_return > 0

        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]
            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                if not self.buffer_created:
                    self.reset(dataset[i][0].shape[0], dataset[i][1].shape[0])
                self._states[self._idx] = dataset[i][0]
                self._actions[self._idx] = dataset[i][1]
                self._rewards[self._idx] = reward

                self._next_states[self._idx] = dataset[i + j][3]
                self._absorbing[self._idx] = dataset[i + j][4]
                self._last[self._idx] = dataset[i + j][5]

                self._idx += 1
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0

                i += 1

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(self._states[i])
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(self._next_states[i])
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last)

    def reset(self, state_dim, action_dim):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.empty((self._max_size, state_dim))
        self._actions = np.empty((self._max_size, action_dim))
        self._rewards = np.empty((self._max_size))
        self._next_states = np.empty((self._max_size, state_dim))
        self._absorbing = np.empty((self._max_size), dtype=bool)
        self._last = np.empty((self._max_size), dtype=bool)
        self.buffer_created = True

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    def _post_load(self):
        if self._full is None:
            self.reset()


class SequenceReplayMemory(ReplayMemory):

    def __init__(self, truncation_length, initial_replay_size, max_replay_size):
        self._truncation_length = truncation_length
        super(SequenceReplayMemory, self).__init__(initial_replay_size, max_replay_size)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()         # previous actions
        aseq = list()       # todo: remove action and use only sequence.
        lengths = list()    # lengths of the sequences

        for i in np.random.randint(self.size, size=n_samples):

            # determine the begin of a sequence
            begin_seq = np.maximum(i - self._truncation_length + 1, 0)
            end_seq = i + 1

            # maybe the sequence contains more than one trajectory, so we need to cut it so that it contains only one
            lasts_absorbing = np.array(self._last[begin_seq-1: i], dtype=int) + \
                              np.array(self._absorbing[begin_seq-1: i], dtype=int)
            begin_traj = np.where(lasts_absorbing > 0)
            sequence_is_shorter_than_requested = len(*begin_traj) > 0
            if sequence_is_shorter_than_requested:
                begin_seq = begin_seq + begin_traj[0][-1]

            # apply padding to the states if needed
            states = np.array(self._states[begin_seq:end_seq])
            next_states = np.array(self._next_states[begin_seq:end_seq])
            action_seq = np.array(self._actions[begin_seq:end_seq])
            if sequence_is_shorter_than_requested or begin_seq == 0:
                prev_actions = np.array(self._actions[begin_seq:end_seq - 1])
                init_prev_action = np.zeros((1, len(self._actions[0])))
                if len(prev_actions) == 0:
                    prev_actions = init_prev_action
                else:
                    prev_actions = np.concatenate([init_prev_action, prev_actions])
            else:
                prev_actions = np.array(self._actions[begin_seq - 1:end_seq - 1])

            length_seq = len(states)

            padded_states = np.concatenate([states, np.zeros((self._truncation_length - states.shape[0],
                                                              states.shape[1]))])
            padded_next_states = np.concatenate([next_states, np.zeros((self._truncation_length - next_states.shape[0],
                                                              next_states.shape[1]))])
            padded_action_seq = np.concatenate([action_seq, np.zeros((self._truncation_length - action_seq.shape[0],
                                                              action_seq.shape[1]))])
            padded_prev_action = np.concatenate([prev_actions, np.zeros((self._truncation_length - prev_actions.shape[0],
                                                              prev_actions.shape[1]))])

            s.append(padded_states)
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(padded_next_states)
            ab.append(self._absorbing[i])
            last.append(self._last[i])
            pa.append(padded_prev_action)
            lengths.append(length_seq)
            aseq.append(padded_action_seq)

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last), np.array(pa), np.array(lengths), np.array(aseq)


class SequenceReplayMemory_with_return(ReplayMemory):

    def __init__(self, truncation_length, initial_size, max_size):
        self._truncation_length = truncation_length
        super().__init__(initial_size, max_size)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()         # previous actions
        aseq = list()       # todo: remove action and use only sequence.
        lengths = list()    # lengths of the sequences
        indices = list()

        for i in np.random.randint(self.size, size=n_samples):

            # determine the begin of a sequence
            begin_seq = np.maximum(i - self._truncation_length + 1, 0)
            end_seq = i + 1

            # maybe the sequence contains more than one trajectory, so we need to cut it so that it contains only one
            lasts_absorbing = np.array(self._last[begin_seq-1: i], dtype=int) + \
                              np.array(self._absorbing[begin_seq-1: i], dtype=int)
            begin_traj = np.where(lasts_absorbing > 0)
            sequence_is_shorter_than_requested = len(*begin_traj) > 0
            if sequence_is_shorter_than_requested:
                begin_seq = begin_seq + begin_traj[0][-1]

            # apply padding to the states if needed
            states = np.array(self._states[begin_seq:end_seq])
            next_states = np.array(self._next_states[begin_seq:end_seq])
            action_seq = np.array(self._actions[begin_seq:end_seq])
            if sequence_is_shorter_than_requested or begin_seq == 0:
                prev_actions = np.array(self._actions[begin_seq:end_seq - 1])
                init_prev_action = np.zeros((1, len(self._actions[0])))
                if len(prev_actions) == 0:
                    prev_actions = init_prev_action
                else:
                    prev_actions = np.concatenate([init_prev_action, prev_actions])
            else:
                prev_actions = np.array(self._actions[begin_seq - 1:end_seq - 1])

            length_seq = len(states)
            padded_states = np.pad(states, [(0, self._truncation_length - length_seq), (0, 0)], mode='constant')
            padded_next_states = np.pad(next_states, [(0, self._truncation_length - len(next_states)),
                                                      (0, 0)], mode='constant')
            padded_action_seq = np.pad(action_seq, [(0, self._truncation_length - len(action_seq)),
                                                      (0, 0)], mode='constant')
            padded_prev_action = np.pad(prev_actions, [(0, self._truncation_length - len(prev_actions)),
                                                       (0, 0)], mode='constant')

            s.append(padded_states)
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(padded_next_states)
            ab.append(self._absorbing[i])
            last.append(self._last[i])
            pa.append(padded_prev_action)
            lengths.append(length_seq)
            aseq.append(padded_action_seq)
            indices.append(i)

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last), np.array(pa), np.array(lengths), np.array(aseq), np.array(indices)

    def set_next_state(self, new_state, ind):
        new_state = new_state.tolist()
        ind = ind.tolist()
        for i, elem in zip(ind, new_state):
            self._next_states[i] = elem
            if not self._absorbing[i] and not self._last[i] and i < self.size - 1:
                self._states[i+1] = elem


class ReplayMemoryPrevAction(ReplayMemory):

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])
            if i == 0 or self._last[i-1] or self._absorbing[i-1]:
                pa.append(np.zeros(len(self._actions[0])))
            else:
                pa.append(self._actions[i-1])
        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last), np.array(pa)


class CorrelatedReplayMemory(ReplayMemory):

    def __init__(self, truncation_length, initial_replay_size, max_replay_size):
        self._truncation_length = truncation_length
        super(CorrelatedReplayMemory, self).__init__(initial_replay_size, max_replay_size)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """

        n_samples = n_samples // self._truncation_length

        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()         # previous actions

        for i in np.random.randint(self.size, size=n_samples):

            # determine the begin of a sequence
            begin_seq = np.maximum(i - self._truncation_length + 1, 0)
            end_seq = i + 1

            # maybe the sequence contains more than one trajectory, so we need to cut it so that it contains only one
            lasts_absorbing = np.array(self._last[begin_seq-1: i], dtype=int) + \
                              np.array(self._absorbing[begin_seq-1: i], dtype=int)
            begin_traj = np.where(lasts_absorbing > 0)
            sequence_is_shorter_than_requested = len(*begin_traj) > 0
            if sequence_is_shorter_than_requested:
                begin_seq = begin_seq + begin_traj[0][-1]

            # apply padding to the states if needed
            states = np.array(self._states[begin_seq:end_seq])
            actions = np.array(self._actions[begin_seq:end_seq])
            next_states = np.array(self._next_states[begin_seq:end_seq])
            rewards = np.array(self._rewards[begin_seq:end_seq])
            absorbings = np.array(self._absorbing[begin_seq:end_seq])
            lasts = np.array(self._last[begin_seq:end_seq])
            if sequence_is_shorter_than_requested or begin_seq == 0:
                prev_actions = np.array(self._actions[begin_seq:end_seq - 1])
                init_prev_action = np.zeros((1, len(self._actions[0])))
                if len(prev_actions) == 0:
                    prev_actions = init_prev_action
                else:
                    prev_actions = np.concatenate([init_prev_action, prev_actions])
            else:
                prev_actions = np.array(self._actions[begin_seq - 1:end_seq - 1])

            s.append(states)
            a.append(actions)
            r.append(rewards)
            ss.append(next_states)
            ab.append(absorbings)
            last.append(lasts)
            pa.append(prev_actions)

        return np.concatenate(s, axis=0), np.concatenate(a, axis=0), np.concatenate(r, axis=0),\
               np.concatenate(ss, axis=0), np.concatenate(ab, axis=0), np.concatenate(last, axis=0),\
               np.concatenate(pa, axis=0)


class ReplayMemoryPrevAction_with_return(ReplayMemory):

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()
        indices = list()

        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])
            indices.append(i)
            if i == 0 or self._last[i-1] or self._absorbing[i-1]:
                pa.append(np.zeros(len(self._actions[0])))
            else:
                pa.append(self._actions[i-1])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last), np.array(pa), np.array(indices)

    def set_next_state(self, new_state, ind):
        new_state = new_state.tolist()
        ind = ind.tolist()
        for i, elem in zip(ind, new_state):
            self._next_states[i] = elem
            if not self._absorbing[i] and not self._last[i] and i < self.size - 1:
                self._states[i+1] = elem


class ReplayMemoryPrevAction_with_returnv2(BaseReplayMemory):

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        pa = list()
        indices = list()

        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])
            indices.append(i)
            if i == 0 or self._last[i-1] or self._absorbing[i-1]:
                pa.append(np.zeros(len(self._actions[0])))
            else:
                pa.append(self._actions[i-1])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last), np.array(pa), np.array(indices)

    def more_efficient_get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """

        ind = np.random.randint(self.size, size=n_samples)
        s = deepcopy(self._states[ind])
        a = deepcopy(self._actions[ind])
        r = deepcopy(self._rewards[ind])
        ss = deepcopy(self._next_states[ind])
        ab = deepcopy(self._absorbing[ind])
        last = deepcopy(self._last[ind])

        pa = deepcopy(a)

        return s, a, r, ss, ab, last, pa, ind

    def set_next_state(self, new_state, ind):
        self._next_states[ind] = new_state
        not_abs = np.where(~self._absorbing[ind], True, False)
        not_last = np.where(~self._last[ind], True, False)
        not_full = np.where(ind<self.size-1, True, False)
        rel_ind = not_abs & not_last & not_full
        ind += 1
        ind = ind[rel_ind]
        self._states[ind] = new_state[rel_ind]


class LSIQReplayMem(ReplayMemory):

    def __init__(self, max_n_expert_traj=10, gamma=0.99, **kwargs):
        self._expert_mem = ExpertMem(max_n_expert_traj)
        self._policy_mem = ReplayMemoryPrevAction_with_return(**kwargs)
        self._gamma = gamma
        del kwargs["mdp_info"]
        super(LSIQReplayMem, self).__init__(**kwargs)

    def add(self, dataset, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert len(dataset) == 1, "This replay memory only supports adding one sample at a time!"

        # add the sample to the replay mem
        self._states[self._idx] = dataset[0][0]
        self._actions[self._idx] = dataset[0][1]
        self._rewards[self._idx] = dataset[0][2]

        self._next_states[self._idx] = dataset[0][3]
        self._absorbing[self._idx] = dataset[0][4]
        self._last[self._idx] = dataset[0][5]

        # if this is the last sample, calculate the J, and maybe add the last trajectory to the expert memory
        if dataset[0][5]:

            begin_traj = 0
            end_traj = self._idx + 1

            last_traj = []
            for i in range(end_traj):
                sample = deepcopy([self._states[i], self._actions[i], self._rewards[i], self._next_states[i],
                          self._absorbing[i], self._last[i]])
                last_traj.append(sample)
                self._states[i] = None
                self._actions[i] = None
                self._rewards[i] = None
                self._next_states[i] = None
                self._absorbing[i] = None
                self._last[i] = None

            #last_traj = self.get_dataset_in_range(begin_traj, end_traj)
            J = compute_J(last_traj, self._gamma)[0]

            # add to expert memory if good enough
            added, replaced_traj = self._expert_mem.add_if_good_enough(last_traj, J)

            if not added:
                self._policy_mem.add(last_traj)
            elif replaced_traj:
                self._policy_mem.add(replaced_traj)

            self._idx = 0

        else:
            self._idx += 1

        if self._idx == self._max_size:
            raise ValueError("Temporary Buffer is smaller then task horion!")

    def get(self, n_samples):
        return self._policy_mem.get(n_samples)

    def get_expert_samples(self, n_samples):
        return self._expert_mem.get(n_samples)

    def get_expert_J_min(self):
        return self._expert_mem.current_min_J()

    def get_expert_J_max(self):
        return self._expert_mem.current_max_J()

    @property
    def initialized(self):
        return self._policy_mem.initialized

    @property
    def size(self):
        return self._policy_mem.size

    def get_dataset_in_range(self, begin_i, end_i):

        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()

        for i in range(begin_i, end_i, 1):
            s.append(self._states[i])
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(self._next_states[i])
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return arrays_as_dataset(np.array(s), np.array(a), np.array(r), np.array(ss), np.array(ab), np.array(last))


class ExpertMem(Serializable):

    def __init__(self, max_n_expert_traj):
        self._max_n_expert_traj = max_n_expert_traj
        self._expert_trajectories = []
        self._expert_trajectories_Js = []

    def reset(self):
        self._expert_trajectories = []
        self._expert_trajectories_Js = []

    def add_if_good_enough(self, traj, J):
        added = False
        traj_to_be_replaced = None
        if len(self._expert_trajectories) < self._max_n_expert_traj:
            self._expert_trajectories.append(deepcopy(traj))
            self._expert_trajectories_Js.append(J)
            added = True
        elif len(self._expert_trajectories) == self._max_n_expert_traj:
            if J > self.current_min_J():
                traj_to_be_replaced = deepcopy(self._expert_trajectories[0])
                self._expert_trajectories[0] = deepcopy(traj)
                self._expert_trajectories_Js[0] = J
                added = True
        else:
            raise ValueError("This should not happen.")

        # sort the buffer
        sorted_ind = np.argsort(self._expert_trajectories_Js)

        new_expert_trajectories = []
        new_expert_trajectories_Js = []

        for i in sorted_ind:
            new_expert_trajectories.append(self._expert_trajectories[i])
            new_expert_trajectories_Js.append(self._expert_trajectories_Js[i])

        self._expert_trajectories = new_expert_trajectories
        self._expert_trajectories_Js = new_expert_trajectories_Js

        return added, traj_to_be_replaced

    def get(self, n_samples):
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        curr_n_exp_traj = len(self._expert_trajectories)
        for i in np.random.randint(curr_n_exp_traj, size=n_samples):
            j = np.random.randint(len(self._expert_trajectories[i]))
            s.append(self._expert_trajectories[i][j][0])
            a.append(self._expert_trajectories[i][j][1])
            r.append(self._expert_trajectories[i][j][2])
            ss.append(self._expert_trajectories[i][j][3])
            ab.append(self._expert_trajectories[i][j][4])
            last.append(self._expert_trajectories[i][j][5])

        return np.array(s), np.array(a), np.array(r), np.array(ss), np.array(ab), np.array(last)

    def current_min_J(self):
        return np.min(self._expert_trajectories_Js)

    def current_max_J(self):
        return np.max(self._expert_trajectories_Js)
