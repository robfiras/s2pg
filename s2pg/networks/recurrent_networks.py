import torch
import numpy as np
from s2pg.cpg import CPGActor
from s2pg.networks.cpg_networks import PPOCPGActor
from s2pg.networks.vanilla_networks import ScaledSigmoid
from s2pg.networks.vanilla_networks import PPONetwork, FullyConnectedNetwork


def get_recurrent_network(rnn_type):
    if rnn_type == "vanilla":
        return torch.nn.RNN
    elif rnn_type == "gru":
        return torch.nn.GRU
    else:
        raise ValueError("Unknown RNN type %s." % rnn_type)


class RecurrentNet(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, n_hidden_features, num_layers, hidden_activation,
                 out_net_params, use_cuda=False, rnn_type="vanilla",
                 pre_net_params=None, **kwargs):

        super().__init__()

        assert len(input_shape) == len(output_shape) == 1
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._num_layers = num_layers
        self._n_hidden_features = n_hidden_features
        self._dim_env_state = dim_env_state
        self._use_cuda = use_cuda
        self._hidden_activation = hidden_activation

        rnn = get_recurrent_network(rnn_type)

        # some things are automatically set
        if pre_net_params:  # todo: I am an idiot, this should be named post_net instead ...
            assert "input_shape" not in pre_net_params.keys()
            assert "output_shape" not in pre_net_params.keys()
        assert "input_shape" not in out_net_params.keys()
        assert "output_shape" not in out_net_params.keys()
        assert "squeeze_out" not in out_net_params.keys()

        if pre_net_params:
            self._pre_net = FullyConnectedNetwork(input_shape=input_shape,
                                                  output_shape=(n_hidden_features,),
                                                  **pre_net_params)
            self._rnn = rnn(input_size=n_hidden_features,
                            hidden_size=n_hidden_features,
                            num_layers=num_layers,
                            #nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                            batch_first=True)
        else:
            self._pre_net = None
            self._rnn = rnn(input_size=input_shape[0],
                            hidden_size=n_hidden_features,
                            num_layers=num_layers,
                            #nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                            batch_first=True)

        self._out_net = FullyConnectedNetwork(input_shape=(n_hidden_features,),
                                              output_shape=output_shape,
                                              squeeze_out=True,
                                              **out_net_params)

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def get_low_high_hidden_state(self):
        if self._hidden_activation == "tanh":
            return -np.ones(self._n_hidden_features*self._num_layers),\
                   np.ones(self._n_hidden_features*self._num_layers)
        elif self._hidden_activation == "relu":
            return np.zeros(self._n_hidden_features*self._num_layers),\
                   np.full(self._n_hidden_features*self._num_layers, np.inf)
        else:
            raise ValueError("Unsupported activation %s." % self._hidden_activation)

    def forward(self, states):
        states = torch.atleast_2d(states)
        env_states, hidden_states = self.divide_state_to_env_hidden_batch(states)

        # expand dim to have sequence length 1 | expected input should have shape (N_batch, L_sequence(=1), DIM_state)
        # todo: for now we only consider sequence lengths of 1, for bptt_pI we need a sequence lengths > 1
        env_states = torch.unsqueeze(env_states, 1)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes
        hidden_states = hidden_states.view(-1, self._num_layers, self._n_hidden_features)
        hidden_states = torch.swapaxes(hidden_states, 0, 1)

        if self._pre_net:
            rnn_input = self._pre_net(env_states)
            rnn_input = torch.unsqueeze(rnn_input, 1)   # todo: for sequence length 1 only
        else:
            rnn_input = env_states

        # forward rnn
        out_rnn, next_hidden_states = self._rnn(rnn_input, hidden_states)

        # forward output network
        out_rnn = torch.squeeze(out_rnn, dim=1)
        action = self._out_net(out_rnn)

        # reshape the hidden states back
        next_hidden_states = torch.swapaxes(next_hidden_states, 0, 1)
        next_hidden_states = next_hidden_states.reshape(-1, self._num_layers * self._n_hidden_features)

        return action, next_hidden_states


class CPGNet(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, cpg_actor_params, post_net_params,
                 dropout=False, use_cuda=False, use_additive_bias=False):
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._cpg_function = CPGActor(**cpg_actor_params)
        self._post_net = PPOCPGActor(**post_net_params)
        self._use_cuda = use_cuda
        self._use_additive_bias = use_additive_bias

    def forward(self, states):
        states = torch.atleast_2d(states)
        env_states, hidden_states = self.divide_state_to_env_hidden_batch(states)

        next_hidden_states = self._cpg_function(hidden_states)

        post_net_inputs = torch.concat([env_states, next_hidden_states], dim=1)
        action, emission = self._post_net(post_net_inputs)
        if self._use_additive_bias:
            out = action + emission
        else:
            out = action
        return out, next_hidden_states

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

class CPGNetAtlas(CPGNet):

    def __init__(self, input_shape, output_shape, dim_env_state, cpg_actor_params, post_net_params,
                 dropout=False, use_cuda=False, use_additive_bias=False):
        super(CPGNetAtlas, self).__init__(input_shape, output_shape, dim_env_state, cpg_actor_params, post_net_params,
                 dropout, use_cuda, use_additive_bias)

        self._ode = self._cpg_function.oscillator_ode
        assert self._ode.n_oscillators == 2

        self._ode.v_max = 3.0
        self._ode.v_min = 0.0
        self._ode.b_max = 2.0
        self._ode.b_min = 0.0
        self._ode.c_max = 10.0
        self._ode.c_min = 0.0
        self._ode.w_max = 1.0
        self._ode.w_min = 0.0
        self._ode.phi_max = 2*np.pi
        self._ode.phi_min = 0.0

        self._ode.v_activation = ScaledSigmoid(self._ode.v_max)
        self._ode.b_activation = ScaledSigmoid(self._ode.b_max)
        self._ode.c_activation = ScaledSigmoid(self._ode.c_max)
        self._ode.w_activation = ScaledSigmoid(self._ode.w_max)
        self._ode.phi_activation = ScaledSigmoid(self._ode.phi_max)

    def forward(self, states):
        states = torch.atleast_2d(states)
        env_states, hidden_states = self.divide_state_to_env_hidden_batch(states)

        next_hidden_states = self._cpg_function(hidden_states)

        post_net_inputs = torch.concat([env_states, next_hidden_states], dim=1)
        action, emission = self._post_net(post_net_inputs)
        batch_size = emission.shape[0]
        # add padding to the emission for the unactivated actuators
        hip_flexion_r = emission[:, 0].view(batch_size, 1)
        hip_abduction_r = torch.zeros_like(emission[:, 0]).view(batch_size, 1)
        hip_rotation_r = torch.zeros_like(emission[:, 0]).view(batch_size, 1)
        knee_r = torch.zeros_like(emission[:, 0]).view(batch_size, 1)
        #ankle_r = emission[:, 1].view(batch_size, 1)
        ankle_r = torch.zeros_like(emission[:, 0]).view(batch_size, 1)

        hip_flexion_l = -emission[:, 0].view(batch_size, 1)
        hip_abduction_l = torch.zeros_like(emission[:, 0]).view(batch_size, 1)
        hip_rotation_l = torch.zeros_like(emission[:, 0]).view(batch_size, 1)
        knee_l = torch.zeros_like(emission[:, 0]).view(batch_size, 1)
        #ankle_l = -emission[:, 1].view(batch_size, 1)
        ankle_l = torch.zeros_like(emission[:, 0]).view(batch_size, 1)

        emission = torch.concat([hip_flexion_r, hip_abduction_r, hip_rotation_r, knee_r, ankle_r, hip_flexion_l,
                                 hip_abduction_l, hip_rotation_l, knee_l, ankle_l], dim=1)

        if self._use_additive_bias:
            out = action + emission
        else:
            out = action
        return out, next_hidden_states


class CPGNetv2(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, cpg_actor_params, post_net_cpg_params,
                 post_net_env_params, psi_mask, r_mask, rd_mask, detach_r_b=False, dropout=False, use_cuda=False):
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._cpg_function = CPGActor(**cpg_actor_params)
        self._post_net_cpg = PPONetwork(**post_net_cpg_params)
        self._post_net_env = PPONetwork(**post_net_env_params)
        self._use_cuda = use_cuda
        self._psi_mask = psi_mask
        self._r_mask = r_mask
        self._rd_mask = rd_mask
        self._detach_r_b = detach_r_b

    def forward(self, states):
        states = torch.atleast_2d(states)
        env_states, hidden_states = self.divide_state_to_env_hidden_batch(states)

        next_hidden_states = self._cpg_function(hidden_states)
        transformed_next_hidden_states = self.transform_cpg_state(next_hidden_states)
        post_net_env_inputs = torch.concat([env_states, transformed_next_hidden_states], dim=1)
        out_env = self._post_net_env(post_net_env_inputs)
        out_cpg = self._post_net_cpg(transformed_next_hidden_states)

        batch_size = states.shape[0]
        r = next_hidden_states[:, self._r_mask]
        b = torch.unsqueeze(self._cpg_function.oscillator_ode.b, dim=0)
        b = torch.tile(b, (batch_size, 1))

        if not self._detach_r_b:
            out = r/b * out_cpg + (1 - r/b) * out_env
        else:
            out = r.detach() / b.detach() * out_cpg + (1 - r.detach() / b.detach()) * out_env

        return out, next_hidden_states

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def transform_cpg_state(self, cpg_state):
        psi = cpg_state[:, self._psi_mask]
        r = cpg_state[:, self._r_mask]
        rd = cpg_state[:, self._rd_mask]
        return torch.concat([r*torch.sin(psi), r*torch.cos(psi), r, rd], dim=1)

class TD3CriticBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1,
                 hidden_state_treatment="zero_initial", **kwargs):
        super().__init__()

        assert hidden_state_treatment in ["zero_initial", "use_policy_hidden_state"]

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._use_policy_hidden_states = True if hidden_state_treatment == "use_policy_hidden_state" else False

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_oa_post_rnn = torch.nn.Linear(dim_env_state+dim_action, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._hq1_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq1_2 = torch.nn.Linear(n_features, 1)
        self._hq2_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq2_2 = torch.nn.Linear(n_features, 1)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_oa_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq1_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq1_2.weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self._hq2_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq2_2.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, state, action, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)

        env_state = env_state.float()
        action = action.float()
        prev_action = prev_action.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # --- forward rnn ---
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)
        if self._use_policy_hidden_states:
            # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
            # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
            first_hidden_state_of_seq = hidden_state[:, 0, :]
            first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers,
                                                                       self._n_hidden_features)
            first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)
            out_rnn, _ = self._rnn(packed_seq, first_hidden_state_of_seq)
        else:
            out_rnn, _ = self._rnn(packed_seq)   # use zero initial states

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        state_action = torch.concat([last_state, action], dim=1)
        feature_sa = self._act_func(self._h1_oa_post_rnn(state_action))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        q1 = self._hq1_2(self._act_func(self._hq1_1(input_last_layer)))
        q2 = self._hq2_2(self._act_func(self._hq2_1(input_last_layer)))

        return torch.concat([q1, q2], dim=1)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class SACHybridCriticBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type, dim_plcy_hidden,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1, **kwargs):
        super().__init__()


        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._dim_plcy_hidden = dim_plcy_hidden

        rnn = get_recurrent_network(rnn_type)

        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_oah_post_rnn = torch.nn.Linear(dim_env_state+dim_action+2*dim_plcy_hidden, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._hq1_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq1_2 = torch.nn.Linear(n_features, 1)
        self._hq2_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq2_2 = torch.nn.Linear(n_features, 1)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_oah_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq1_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq1_2.weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self._hq2_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq2_2.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, state, action, next_hidden, prev_action, lengths, use_hidden_states=False,
                return_hidden_states=False):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)

        env_state = env_state.float()
        action = action.float()
        prev_action = prev_action.float()
        hidden_state = hidden_state.float()
        next_hidden = next_hidden.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # --- forward rnn ---
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)
        if use_hidden_states:
            # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
            # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
            first_hidden_state_of_seq = hidden_state[:, 0, :]
            first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers,
                                                                       self._n_hidden_features)
            first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)
            out_rnn, next_next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)
        else:
            out_rnn, next_next_hidden = self._rnn(packed_seq)   # use zero initial states

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        last_hidden_state = torch.squeeze(torch.take_along_dim(hidden_state, rel_indices, dim=1), dim=1)
        state_action_hidden = torch.concat([last_state, action, last_hidden_state, next_hidden], dim=1)
        feature_sa = self._act_func(self._h1_oah_post_rnn(state_action_hidden))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        q1 = self._hq1_2(self._act_func(self._hq1_1(input_last_layer)))
        q2 = self._hq2_2(self._act_func(self._hq2_1(input_last_layer)))

        if return_hidden_states:
            return torch.concat([q1, q2], dim=1), next_next_hidden
        else:
            return torch.concat([q1, q2], dim=1)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class SACHybridCriticBPTTNetwork_QStates(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type, dim_plcy_hidden,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1, **kwargs):
        super().__init__()


        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._dim_plcy_hidden = dim_plcy_hidden
        self._n_features = n_features

        rnn = get_recurrent_network(rnn_type)

        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_oa_post_rnn = torch.nn.Linear(dim_env_state+dim_action, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._hq1_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq1_2 = torch.nn.Linear(n_features, 1)
        self._hq2_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq2_2 = torch.nn.Linear(n_features, 1)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_oa_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq1_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq1_2.weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self._hq2_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq2_2.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, state, action, prev_action, lengths, use_hidden_states=False,
                return_hidden_states=False):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)

        env_state = env_state.float()
        action = action.float()
        prev_action = prev_action.float()
        hidden_state = hidden_state.float()

        if type(use_hidden_states) != bool:
            assert torch.all(use_hidden_states) or torch.all(~use_hidden_states)
            use_hidden_states = use_hidden_states[0]

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # --- forward rnn ---
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)
        if use_hidden_states:
            # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
            # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
            first_hidden_state_of_seq = hidden_state[:, 0, :]
            first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers,
                                                                       self._n_hidden_features)
            first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)
            out_rnn, next_next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)
        else:
            out_rnn, next_next_hidden = self._rnn(packed_seq)   # use zero initial states

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        state_action_hidden = torch.concat([last_state, action], dim=1)
        feature_sa = self._act_func(self._h1_oa_post_rnn(state_action_hidden))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        q1 = self._hq1_2(self._act_func(self._hq1_1(input_last_layer)))
        q2 = self._hq2_2(self._act_func(self._hq2_1(input_last_layer)))

        if return_hidden_states:
            return torch.concat([q1, q2], dim=1), next_next_hidden
        else:
            return torch.concat([q1, q2], dim=1)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]

class SAC_BPPT_Stochastic_Net(SACHybridCriticBPTTNetwork_QStates):


    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type, dim_plcy_hidden,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1, **kwargs):
        super().__init__(input_shape, output_shape, dim_env_state, dim_action, rnn_type, dim_plcy_hidden,
                 n_hidden_features, n_features, num_hidden_layers, **kwargs)
        self._h_log_sigma = torch.nn.Linear(self._n_features + self._n_hidden_features, self._n_hidden_features)
        torch.nn.init.xavier_uniform_(self._h_log_sigma.weight, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, state, action, prev_action, lengths, use_hidden_states=False, return_hidden_states=True):

        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)

        env_state = env_state.float()
        action = action.float()
        prev_action = prev_action.float()
        hidden_state = hidden_state.float()

        if type(use_hidden_states) != bool:
            assert torch.all(use_hidden_states) or torch.all(~use_hidden_states)
            use_hidden_states = use_hidden_states[0]

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # --- forward rnn ---
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)
        if use_hidden_states:
            # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
            # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
            first_hidden_state_of_seq = hidden_state[:, 0, :]
            first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers,
                                                                       self._n_hidden_features)
            first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)
            out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)
        else:
            out_rnn, next_hidden = self._rnn(packed_seq)  # use zero initial states

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        state_action_hidden = torch.concat([last_state, action], dim=1)
        feature_sa = self._act_func(self._h1_oa_post_rnn(state_action_hidden))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        q1 = self._hq1_2(self._act_func(self._hq1_1(input_last_layer)))
        q2 = self._hq2_2(self._act_func(self._hq2_1(input_last_layer)))

        if return_hidden_states:
            # compute the log_sigma
            log_sigma_in = torch.concat([feature_sa, torch.squeeze(next_hidden, dim=0)], dim=1)
            log_sigma = self._h_log_sigma(log_sigma_in)

            # draw the next_hidden state from gaussian and squash
            dist = torch.distributions.Normal(torch.squeeze(next_hidden, dim=0), log_sigma.exp())
            stoch_next_hidden = dist.rsample()
            stoch_next_hidden = torch.tanh(stoch_next_hidden)

            return torch.concat([q1, q2], dim=1), stoch_next_hidden
        else:
            return torch.concat([q1, q2], dim=1)


class TD3ActorBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, action_scaling, dim_env_state, rnn_type,
                 n_hidden_features, num_hidden_layers=1, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = output_shape[0]
        n_output = output_shape[0]
        self._dim_env_state = dim_env_state
        self._action_scaling = action_scaling
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_hidden_features+n_features, n_output)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("linear")*0.1)

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        raw_a = self._h3(input_last_layer)

        return self._action_scaling * self._tanh(raw_a), next_hidden

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class PPOCriticBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1,
                 hidden_state_treatment="zero_initial", **kwargs):
        super().__init__()

        assert hidden_state_treatment in ["zero_initial", "use_policy_hidden_state"]

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._use_policy_hidden_states = True if hidden_state_treatment == "use_policy_hidden_state" else False

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_env_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._hq_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq_2 = torch.nn.Linear(n_features, 1)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq_2.weight, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)

        env_state = env_state.float()
        prev_action = prev_action.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # --- forward rnn ---
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)
        if self._use_policy_hidden_states:
            # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
            # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
            first_hidden_state_of_seq = hidden_state[:, 0, :]
            first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers,
                                                                       self._n_hidden_features)
            first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)
            out_rnn, _ = self._rnn(packed_seq, first_hidden_state_of_seq)
        else:
            out_rnn, _ = self._rnn(packed_seq)   # use zero initial states

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_s = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_s, features_rnn], dim=1)
        q = self._hq_2(self._act_func(self._hq_1(input_last_layer)))

        return torch.squeeze(q)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class PPOActorBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, rnn_type, dim_action,
                 n_hidden_features, num_hidden_layers=1, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_env_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_hidden_features+n_features, dim_action)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        a = self._h3(input_last_layer)

        return a, torch.swapaxes(next_hidden, 0, 1)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]

class TD3ActorRecurrentNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, action_scaling, dim_env_state, rnn_type,
                 n_hidden_features, num_hidden_layers=1, hidden_activation="tanh", rate_change_hidden_state=1.0,
                 use_prev_action=True, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = output_shape[0]
        n_output = output_shape[0]
        self._dim_env_state = dim_env_state
        self._action_scaling = action_scaling
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features
        assert hidden_activation == "tanh"              # todo: currently only tanh is supported
        self._hidden_activation = hidden_activation
        self._rate_change_hidden_state = rate_change_hidden_state
        self._use_prev_action = use_prev_action

        rnn = get_recurrent_network(rnn_type)

        self.detach_hidden = False

        # embedder
        self._h1_o = torch.nn.Linear(dim_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        rnn_input_size = n_features*2 if self._use_prev_action else n_features
        self._rnn = rnn(input_size=rnn_input_size,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        self._hidden_state_multiplier = 1

        # post-rnn layer
        self._h3 = torch.nn.Linear(num_hidden_layers*n_hidden_features+n_features, n_output)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("linear")*0.1)


    def forward(self, state, prev_action):
        state = torch.atleast_2d(state)
        prev_action = torch.atleast_2d(prev_action)
        lengths = torch.ones(state.shape[0], dtype=torch.long)
        state = torch.unsqueeze(state, dim=1)
        prev_action = torch.unsqueeze(prev_action, dim=1)
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        if self._use_prev_action:
            input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        else:
            input_rnn = feature_o

        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]

        # feature
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        next_hidden = (1-self._rate_change_hidden_state) * first_hidden_state_of_seq + \
                      self._rate_change_hidden_state * next_hidden

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        if self.detach_hidden:
            features_rnn = features_rnn.detach()
            next_hidden = next_hidden.detach()

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        raw_a = torch.squeeze(self._h3(input_last_layer))

        return self._action_scaling * self._tanh(raw_a), torch.squeeze(next_hidden)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def get_low_high_hidden_state(self):
        if self._hidden_activation == "tanh":
            return -np.ones(self._n_hidden_features*self._num_hidden_layers) * self._hidden_state_multiplier,\
                   np.ones(self._n_hidden_features*self._num_hidden_layers) * self._hidden_state_multiplier
        elif self._hidden_activation == "relu":
            return np.zeros(self._n_hidden_features*self._num_hidden_layers),\
                   np.full(self._n_hidden_features*self._num_hidden_layers, np.inf)
        else:
            raise ValueError("Unsupported activation %s." % self._hidden_activation)


class TD3WindowActorNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, action_scaling, window_length,
                 dim_env_state, use_prev_action=False, **kwargs):

        super(TD3WindowActorNetwork, self).__init__()

        self._action_scaling = action_scaling
        self._use_prev_action = use_prev_action
        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]
        self._dim_env_state = dim_env_state
        self._window_length = window_length

        if use_prev_action:
            self._h1 = torch.nn.Linear(self._dim_env_state * window_length + dim_action * window_length, n_features)
        else:
            self._h1 = torch.nn.Linear(self._dim_env_state * window_length, n_features)
        self._h2 = torch.nn.Linear(n_features, n_features)
        self._out = torch.nn.Linear(n_features, self.output_shape)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._h1.weight, gain=torch.nn.init.calculate_gain("relu") * 0.1)
        torch.nn.init.xavier_uniform_(self._h2.weight, gain=torch.nn.init.calculate_gain("relu") * 0.1)
        torch.nn.init.xavier_uniform_(self._out.weight, gain=torch.nn.init.calculate_gain("relu") * 0.1)

    def forward(self, state, prev_action, lengths=None):

        env_state, _ = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # concatenate all states
        env_state = torch.flatten(env_state, start_dim=1)

        if self._use_prev_action:
            prev_action = torch.flatten(prev_action, start_dim=1)
            env_state = torch.concat([env_state, prev_action], dim=1)

        z = self._act_func(self._h1(env_state))
        z = self._act_func(self._h2(z))
        raw_a = self._out(z)
        raw_a = torch.squeeze(raw_a)

        return self._action_scaling * self._tanh(raw_a), None

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class PPOActorRecurrentNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, rnn_type, action_dim,
                 n_hidden_features, num_hidden_layers=1, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = action_dim
        n_output = output_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_hidden_features+n_features, dim_action)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        a = self._h3(input_last_layer)

        return a, torch.swapaxes(next_hidden, 0, 1)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def get_low_high_hidden_state(self):
        if self._hidden_activation == "tanh":
            return -np.ones(self._n_hidden_features*self._num_hidden_layers),\
                   np.ones(self._n_hidden_features*self._num_hidden_layers)
        elif self._hidden_activation == "relu":
            return np.zeros(self._n_hidden_features*self._num_hidden_layers),\
                   np.full(self._n_hidden_features*self._num_hidden_layers, np.inf)
        else:
            raise ValueError("Unsupported activation %s." % self._hidden_activation)


class SACRecurrentActorNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, dim_action, rnn_type,
                 n_hidden_features, num_hidden_layers=1,  rate_change_hidden_state=1.0, use_prev_action=True,
                 action_use_next_hidden=True, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        assert output_shape[0] % 2 == 0
        dim_action = dim_action
        n_output = output_shape[0]
        self._dim_action = dim_action
        self.output_shape = output_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features
        self._rate_change_hidden_state = rate_change_hidden_state
        self._use_prev_action = use_prev_action
        self._action_use_next_hidden = action_use_next_hidden

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(self._dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(self._dim_env_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        rnn_input_size = n_features*2 if self._use_prev_action else n_features
        self._rnn = rnn(input_size=rnn_input_size,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._h4 = torch.nn.Linear(n_features, n_output)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h4.weight, gain=torch.nn.init.calculate_gain("linear")*0.1)

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        if self._use_prev_action:
            input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        else:
            input_rnn = feature_o
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        next_hidden = (1-self._rate_change_hidden_state) * first_hidden_state_of_seq + \
                      self._rate_change_hidden_state * next_hidden

        # we only need the last entry in each sequence
        if self._action_use_next_hidden:
            features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
            rel_indices = lengths.view(-1, 1, 1) - 1
            features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)
        else:
            rel_indices = lengths.view(-1, 1, 1) - 1
            features_rnn = hidden_state[:, 0, :]

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        raw_out = self._h4(self._act_func(self._h3(input_last_layer)))
        mu_action = raw_out[:, :self._dim_action]
        log_sigma_action = raw_out[:, self._dim_action:2*self._dim_action]
        log_sigma_next_hidden_state = raw_out[:, 2*self._dim_action:]

        mus = torch.concat([mu_action, torch.squeeze(next_hidden, dim=0)], dim=1)   # todo: concat accross first dim instead to work with n layer rnns!
        log_sigmas = torch.concat([log_sigma_action, log_sigma_next_hidden_state], dim=1)

        return mus, log_sigmas

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class SACRecurrentActorNetwork_Q(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, dim_action, rnn_type,
                 n_hidden_features, num_hidden_layers=1,  rate_change_hidden_state=1.0, use_prev_action=True,
                 action_use_next_hidden=True, **kwargs):

        super().__init__()

        dim_state = input_shape[0]
        assert output_shape[0] % 2 == 0
        dim_action = dim_action
        n_output = output_shape[0]
        self._dim_action = dim_action
        self.output_shape = output_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features
        self._rate_change_hidden_state = rate_change_hidden_state
        self._use_prev_action = use_prev_action
        self._action_use_next_hidden = action_use_next_hidden

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_state, n_features)
        self._h1_pi = torch.nn.Linear(dim_state+dim_action+num_hidden_layers*n_hidden_features, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        rnn_input_size = n_features*2 if self._use_prev_action else n_features
        self._rnn = rnn(input_size=rnn_input_size,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_features, n_features//2)
        self._h4 = torch.nn.Linear(n_features//2, 2*dim_action)
        self._h5 = torch.nn.Linear(n_hidden_features + dim_env_state + dim_action, n_features//2)
        self._h6 = torch.nn.Linear(n_features//2, n_hidden_features)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_pi.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h4.weight, gain=torch.nn.init.calculate_gain("linear")*0.1)
        torch.nn.init.xavier_uniform_(self._h5.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h6.weight, gain=torch.nn.init.calculate_gain("linear")*0.1)

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        if self._use_prev_action:
            input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        else:
            input_rnn = feature_o
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        next_hidden = (1-self._rate_change_hidden_state) * first_hidden_state_of_seq + \
                      self._rate_change_hidden_state * next_hidden

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1).detach()

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        last_prev_action = torch.squeeze(torch.take_along_dim(prev_action, rel_indices, dim=1), dim=1)
        last_hidden_state = torch.squeeze(torch.take_along_dim(hidden_state, rel_indices, dim=1), dim=1)
        feature_sah = self._act_func(self._h1_pi(torch.cat([last_state, last_hidden_state, last_prev_action], dim=1)))

        # last layer
        mu_logsigma_action = self._h4(self._act_func(self._h3(feature_sah)))
        mu_action = mu_logsigma_action[:, :self._dim_action]
        log_sigma_action = mu_logsigma_action[:, self._dim_action:]
        log_sigma_next_hidden_state = self._h6(self._act_func(self._h5(torch.concat([features_rnn,
                                                                                     last_state,
                                                                                     last_prev_action], dim=1))))

        mus = torch.concat([mu_action, torch.squeeze(next_hidden, dim=0)], dim=1)   # todo: concat accross first dim instead to work with n layer rnns!
        log_sigmas = torch.concat([log_sigma_action, log_sigma_next_hidden_state], dim=1)

        return mus, log_sigmas

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class SACBPTTActorNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, dim_action, rnn_type,
                 n_hidden_features, num_hidden_layers=1,  rate_change_hidden_state=1.0, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        assert output_shape[0] % 2 == 0
        dim_action = dim_action
        n_output = output_shape[0]
        self._dim_action = dim_action
        self.output_shape = output_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features
        self._rate_change_hidden_state = rate_change_hidden_state

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_state, n_features)
        self._h1_a_prev = torch.nn.Linear(dim_action, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features*2,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._h4 = torch.nn.Linear(n_features, n_output)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h1_a_prev.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h4.weight, gain=torch.nn.init.calculate_gain("linear")*0.1)

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # pre-rnn embedder
        feature_o = self._act_func(self._h1_o(env_state))
        feature_a_prev = self._act_func(self._h1_a_prev(prev_action))

        # forward rnn
        input_rnn = torch.concat([feature_o, feature_a_prev], dim=2)
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._n_hidden_features)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        next_hidden = (1-self._rate_change_hidden_state) * first_hidden_state_of_seq + \
                      self._rate_change_hidden_state * next_hidden

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(env_state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        raw_out = self._h4(self._act_func(self._h3(input_last_layer)))
        mu_action = raw_out[:, :self._dim_action]
        log_sigma_action = raw_out[:, self._dim_action:]

        return mu_action, log_sigma_action, next_hidden

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class SACWindowActorNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, dim_action, window_length=5,
                 use_prev_action=False, **kwargs):

        super().__init__()

        assert output_shape[0] % 2 == 0

        self._use_prev_action = use_prev_action
        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._window_length = window_length

        if use_prev_action:
            self._h1 = torch.nn.Linear(self._dim_env_state * window_length + dim_action * window_length, n_features)
        else:
            self._h1 = torch.nn.Linear(self._dim_env_state * window_length, n_features)
        self._h2 = torch.nn.Linear(n_features, n_features)
        self._out = torch.nn.Linear(n_features, self.output_shape)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._h2.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)
        torch.nn.init.xavier_uniform_(self._out.weight, gain=torch.nn.init.calculate_gain("relu")*0.1)

    def forward(self, state, prev_action, lengths):

        env_state, _ = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # concatenate all states
        env_state = torch.flatten(env_state, start_dim=1)

        if self._use_prev_action:
            prev_action = ttorch.flatten(prev_action, start_dim=1)
            env_state = torch.concat([env_state, prev_action], dim=1)

        z = self._act_func(self._h1(env_state))
        z = self._act_func(self._h2(z))
        out = self._out(z)

        mu_action = out[:, :self._dim_action]
        log_sigma_action = out[:, self._dim_action:]

        return mu_action, log_sigma_action

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class RecurrentREINFORCE(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state,
                 dim_hidden_state, dim_action, use_prev_action=False, use_cuda=False, **kwargs):
        super().__init__()

        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]
        self._dim_env_state = dim_env_state
        self._dim_hidden_state = dim_hidden_state
        self._dim_action = dim_action
        self._use_prev_action = use_prev_action
        self._num_hidden_layers = 1

        vanilla = get_recurrent_network("vanilla")
        self._rnn = vanilla(input_size=n_features,
                            hidden_size=dim_hidden_state,
                            num_layers=self._num_hidden_layers,
                            nonlinearity="relu",
                            batch_first=True)

        # layers
        dim_input = dim_env_state+dim_action if use_prev_action else dim_env_state
        self._h1 = torch.nn.Linear(dim_input, n_features)
        self._h3 = torch.nn.Linear(n_features, dim_action)
        self._act_func = torch.nn.Tanh()

    def forward(self, state, prev_action, lengths):
        env_state, hidden_state = self.divide_state_to_env_hidden_batch_seq(state)
        env_state = env_state.float()

        # forward rnn
        if self._use_prev_action:
            input_h1 = torch.concat([env_state, prev_action], dim=2)
            input_rnn = self._act_func(self._h1(input_h1))
        else:
            input_rnn = self._act_func(self._h1(env_state))

        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes. And we only need the first hidden state of sequence.
        first_hidden_state_of_seq = hidden_state[:, 0, :]
        first_hidden_state_of_seq = first_hidden_state_of_seq.view(-1, self._num_hidden_layers, self._dim_hidden_state)
        first_hidden_state_of_seq = torch.swapaxes(first_hidden_state_of_seq, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, first_hidden_state_of_seq)

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # pass to the output layer
        mus = torch.concat([self._h3(features_rnn), features_rnn], dim=1)
        return mus

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]