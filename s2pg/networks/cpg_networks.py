from itertools import chain

import numpy as np
import torch
from mushroom_rl.utils.torch import set_weights
from torch import nn as nn

from .vanilla_networks import FullyConnectedNetwork, TD3ActorNetwork, PPONetwork


class EmissionExtractorFunc(torch.nn.Module):

    def __init__(self, input_shape, output_shape, psi_mask, r_mask, **kwargs):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._psi_mask = psi_mask
        self._r_mask = r_mask

        super().__init__()

        self._bias = nn.Parameter(torch.zeros(output_shape), requires_grad=True)

    def forward(self, cpg_state):
        assert len(cpg_state.shape) >= 2, "Expected at least a 2D cpg state."
        psi = cpg_state[:, self._psi_mask]
        r = cpg_state[:, self._r_mask]
        e = r * torch.cos(psi)
        bias = torch.tile(self._bias, (len(e), 1))
        assert e.shape == bias.shape, "Emission shape %s is unequal bias shape %s." % (e.shape, self._bias.shape)
        return e + bias


class EmissionExtractorLinearLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape, psi_mask, r_mask, use_cuda=False, **kwargs):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._psi_mask = psi_mask
        self._r_mask = r_mask
        self._use_cuda = use_cuda

        super().__init__()

        dim_cpg_state = 3
        assert input_shape[0] % dim_cpg_state == 0
        self._linear = nn.Linear(input_shape[0]//dim_cpg_state, output_shape[0])

        with torch.no_grad():
            self._linear.bias.normal_(std=0.01)
            self._linear.weight.normal_(std=0.05, mean=1.0)

        # do a custom initialization
        n_cpgs = input_shape[0]//dim_cpg_state
        if n_cpgs == output_shape[0]:
            w = np.ones((n_cpgs, output_shape[0]))
            idx = np.where(~np.eye(w.shape[0], dtype=bool))
            w[idx] = np.random.normal(0.0, 0.0, w[idx].size)
            bias = np.random.normal(0.0, 0.0, output_shape)
            w = np.concatenate([w.flatten(), bias])
            set_weights(self._linear.parameters(), w, use_cuda=self._use_cuda)

            self._linear.requires_grad_(False) # todo: this works well in practice, check for non-square case!
        else:
            nn.init.xavier_uniform_(self._linear.weight,
                                    gain=nn.init.calculate_gain("linear"))

    def forward(self, cpg_state):
        assert len(cpg_state.shape) >= 2, "Expected a 2D cpg state."
        assert cpg_state.shape[1] == self._input_shape[0], "Expected cpg state to have dim %d, but got %d." %\
                                                           (cpg_state.shape[1], self._input_shape[0])
        cpg_state = cpg_state.type(torch.float32)
        psi = cpg_state[:, self._psi_mask]
        r = cpg_state[:, self._r_mask]
        e = r * torch.cos(psi)
        out = self._linear(e)
        return out


class CombPostPolicy(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, network_params, emission_extractor_params,
                 use_additive_emission=False, use_cuda=False, **kwargs):

        # call base constructor
        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape

        # we use a linear emission extractor
        self._emission_extractor = EmissionExtractorLinearLayer(use_cuda=use_cuda, **emission_extractor_params)

        # some fully connected network for state-dependent post policy
        self._network = FullyConnectedNetwork(use_cuda=use_cuda, **network_params)

        self._dim_env_state = dim_env_state

        self._use_cuda = use_cuda
        self._use_additive_emission = use_additive_emission

    def forward(self, states):
        assert len(states.shape) == 2, "Expected a 2D state."
        env_state, cpg_states = self.divide_state_to_env_cpg_batch(states)

        emission_output = self._emission_extractor(cpg_states)
        net_input = torch.concat([env_state, emission_output], dim=1)
        network_output = self._network(net_input)
        if self._use_additive_emission:
            action = network_output + emission_output
        else:
            action = network_output
        return torch.tanh(action)

    def parameters(self, recurse: bool = True):
        return self._network.parameters()

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]


class TD3CPGActor(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, network_params, emission_extractor_params,
                 use_cuda=False, use_additive_bias=False, **kwargs):

        # call base constructor
        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape

        # we use a linear emission extractor
        self._emission_extractor = EmissionExtractorLinearLayer(use_cuda=use_cuda, **emission_extractor_params)

        # some fully connected network for state-dependent post policy
        self._use_additive_bias = use_additive_bias
        if use_additive_bias:
            self._action_scaling = network_params["action_scaling"]
            self._network = TD3ActorNetwork(use_cuda=use_cuda, squash_and_scale_action=False, **network_params)
        else:
            self._network = TD3ActorNetwork(use_cuda=use_cuda, squash_and_scale_action=True, **network_params)

        self._dim_env_state = dim_env_state

        self._use_cuda = use_cuda

    def forward(self, states):
        assert len(states.shape) == 2, "Expected a 2D state."
        env_state, cpg_states = self.divide_state_to_env_cpg_batch(states)

        emission_output = self._emission_extractor(cpg_states)
        #net_input = torch.concat([env_state, emission_output], dim=1)
        action = self._network(emission_output)

        if self._use_additive_bias:
            # add bias, then squash and scale action
            action = action + emission_output
            return self._action_scaling * torch.tanh(action)
        else:
            # squashing and scaling already done, we can directly return the action
            return action

    def parameters(self, recurse: bool = True):
        return self._network.parameters()

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]


class TD3CPGCritic(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, state_dim, action_dim, hidden_state_dim, dim_env_state,
                 psi_mask, r_mask, rd_mask, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state
        self._n_features = n_features
        self._psi_mask = psi_mask
        self._r_mask = r_mask
        self._rd_mask = rd_mask

        # Assume there are two hidden layers
        assert len(n_features) == 2, 'TD3 critic needs 2 hidden layers'

        self._h1 = nn.Linear(state_dim + action_dim + hidden_state_dim, n_features[0])
        self._h2_sax = nn.Linear(n_features[0], n_features[1])
        self._h2_a = nn.Linear(action_dim, n_features[1], bias=False)
        self._h2_xn = nn.Linear(hidden_state_dim, n_features[1], bias=False)
        self._h3 = nn.Linear(n_features[1], output_shape[0])
        self._act_func = torch.nn.ReLU()

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2_sax, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_sax.weight)
        nn.init.uniform_(self._h2_sax.weight, a=-1 / np.sqrt(fan_in_h2_sax), b=1 / np.sqrt(fan_in_h2_sax))

        fan_in_h2_a, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_a.weight)
        nn.init.uniform_(self._h2_a.weight, a=-1 / np.sqrt(fan_in_h2_a), b=1 / np.sqrt(fan_in_h2_a))

        fan_in_h2_xn, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_xn.weight)
        nn.init.uniform_(self._h2_xn.weight, a=-1 / np.sqrt(fan_in_h2_xn), b=1 / np.sqrt(fan_in_h2_xn))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state_hidden_state, action, next_hidden_state):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]
        hidden_state = self.transform_cpg_state(hidden_state).float()
        next_hidden_state = self.transform_cpg_state(next_hidden_state).float()
        state_action_hidden_state = torch.cat((env_state, action, hidden_state), dim=1).float()
        action = action.float()

        features1 = self._act_func(self._h1(state_action_hidden_state))
        features2_s = self._h2_sax(features1)
        features2_a = self._h2_a(action)
        features2_xn = self._h2_xn(next_hidden_state)
        features2 = self._act_func(features2_s + features2_a + features2_xn)

        q = self._h3(features2)
        return torch.squeeze(q)

    def transform_cpg_state(self, cpg_state):
        psi = cpg_state[:, self._psi_mask]
        r = cpg_state[:, self._r_mask]
        rd = cpg_state[:, self._rd_mask]
        return torch.concat([r*torch.sin(psi), r*torch.cos(psi), rd], dim=1)


class PPOCPGActor(torch.nn.Module):

    def __init__(self, dim_env_state, network_params, emission_extractor_params,
                 use_cuda=False, **kwargs):

        # call base constructor
        super().__init__()

        # we use a linear emission extractor
        self._emission_extractor = EmissionExtractorLinearLayer(use_cuda=use_cuda, **emission_extractor_params)

        # some fully connected network for state-dependent post policy
        self._network = PPONetwork(use_cuda=use_cuda, **network_params)

        self._dim_env_state = dim_env_state

        self._use_cuda = use_cuda

    def forward(self, states):
        assert len(states.shape) == 2, "Expected a 2D state."
        env_state, cpg_states = self.divide_state_to_env_cpg_batch(states)

        emission_output = self._emission_extractor(cpg_states)
        net_input = torch.concat([env_state, emission_output], dim=1)
        action = self._network(net_input)
        return action, emission_output

    def parameters(self, recurse: bool = True):
        return self._network.parameters()

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]


class NonLinearPostPolicy(CombPostPolicy):

    def __init__(self, input_shape, output_shape, dim_env_state, network_params, emission_extractor_params,
                 use_additive_emission=False, use_cuda=False, **kwargs):

        # call base constructor
        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape

        # we use a linear emission extractor
        self._emission_extractor = EmissionExtractorLinearLayer(use_cuda=use_cuda, **emission_extractor_params)

        # some fully connected network for state-depenedent post policy
        self._network = FullyConnectedNetwork(use_cuda=use_cuda, **network_params)

        self._dim_env_state = dim_env_state

        self._use_cuda = use_cuda
        self._use_additive_emission = use_additive_emission

    def forward(self, states):
        assert len(states.shape) == 2, "Expected a 2D state."
        emission_output = self._emission_extractor(states)
        network_output = self._network(emission_output)
        action = network_output
        return torch.tanh(action)


class TwoBranchNonLinear(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, env_network_params, cpg_network_params, emission_extractor_params,
                 use_action_in_env_net=False, use_cuda=False, **kwargs):

        # call base constructor
        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape

        # we use a linear emission extractor
        self._emission_extractor = EmissionExtractorLinearLayer(use_cuda=use_cuda, **emission_extractor_params)

        # some fully connected network for state-depenedent post policy
        self._network_env = FullyConnectedNetwork(use_cuda=use_cuda, **env_network_params)
        self._network_cpg = FullyConnectedNetwork(use_cuda=use_cuda, **cpg_network_params)

        self._dim_env_state = dim_env_state

        self._use_cuda = use_cuda
        self._use_action_in_env_net = use_action_in_env_net

    def forward(self, states):
        assert len(states.shape) == 2, "Expected a 2D state."
        env_state, cpg_states = self.divide_state_to_env_cpg_batch(states)

        emission_output = self._emission_extractor(cpg_states)
        cpg_action = self._network_cpg(emission_output)

        if self._use_action_in_env_net:
            net_input = torch.concat([env_state, cpg_action], dim=1)
        else:
            net_input = env_state

        env_net_action = self._network_env(net_input)

        return torch.tanh(cpg_action + env_net_action)

    def parameters(self, recurse: bool = True):
        return chain(self._network_env.parameters(), self._network_cpg.parameters())

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]


class TransformedCritic(FullyConnectedNetwork):

    def __init__(self, input_shape, output_shape, dim_env_state, psi_mask, r_mask, rd_mask, **kwargs):
        self._dim_env_state = dim_env_state
        self._psi_mask = psi_mask
        self._r_mask = r_mask
        self._rd_mask = rd_mask
        super().__init__(input_shape, output_shape, **kwargs)

    def divide_state_to_env_cpg_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def forward(self, *inputs):
        states = inputs[0]
        actions = inputs[1]
        next_cpg_states = inputs[2]
        env_states, cpg_states = self.divide_state_to_env_cpg_batch(states)

        cpg_states = self.transform_cpg_state(cpg_states)
        next_cpg_states = self.transform_cpg_state(next_cpg_states)

        return super(TransformedCritic, self).forward(env_states, cpg_states, actions, next_cpg_states)

    def transform_cpg_state(self, cpg_state):
        psi = cpg_state[:, self._psi_mask]
        r = cpg_state[:, self._r_mask]
        rd = cpg_state[:, self._rd_mask]
        return torch.concat([r*torch.sin(psi), r*torch.cos(psi), rd], dim=1)


class TransformedValueCritic(TransformedCritic):

    def forward(self, *inputs):
        env_states, cpg_states = self.divide_state_to_env_cpg_batch(*inputs)

        cpg_states = self.transform_cpg_state(cpg_states)

        # call grandparent
        return super(TransformedCritic, self).forward(env_states, cpg_states)