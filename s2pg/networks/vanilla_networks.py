import torch
import numpy as np
from torch import nn as nn
from torch.nn import ModuleList


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, n_features, activations, activations_params=None,
                 initializers=None, squeeze_out=False, standardizer=None, use_cuda=False, **kwargs):
        """
        This class implements a simple fully-connected feedforward network using torch.
        Args:
            input_shape (Tuple): Shape of the input (only 1-Dim) allowed.
            output_shape (Tuple): Shape of the output (only 1-Dim) allowed.
            n_features (List): Number of dimensions of the hidden layers,
            activations (List): List containing the activation names for each layer.
                                NOTE: len(dims_layers)-1 = len(activations)
            activations_params (List): List of dicts containing the parameters for the activations for each layer.

        """

        # call base constructor
        super().__init__()

        assert len(input_shape) == len(output_shape) == 1

        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]
        dims_network = [self.input_shape] + n_features + [self.output_shape]
        assert len(activations) == len(dims_network) - 1
        assert len(activations) == len(activations_params) if activations_params is not None else True

        # construct the linear layers
        self._linears = ModuleList([nn.Linear(dims_network[i], dims_network[i+1]) for i in range(len(dims_network)-1)])

        # add activations
        if activations_params:
            self._activations = ModuleList([self.activation_function(name, params) for name, params in zip(activations, activations_params)])
        else:
            self._activations = ModuleList([self.activation_function(name) for name in activations])

        self._stand = standardizer
        self._squeeze_out = squeeze_out

        # make initialization
        if initializers is None:
            for layer, activation in zip(self._linears, activations):
               try:
                   nn.init.xavier_uniform_(layer.weight,
                                           gain=nn.init.calculate_gain(activation))
               except:
                   nn.init.xavier_uniform_(layer.weight, gain=0.1)
        else:
            for layer, initializer in zip(self._linears, initializers):
                initializer(layer.weight)

    def forward(self, *inputs, dim=1):
        inputs = torch.squeeze(torch.cat(inputs, dim=dim), 1)
        if len(inputs.shape) == 1:
            inputs = torch.unsqueeze(inputs, dim=1)
        if self._stand is not None:
            inputs = self._stand(inputs)
        # define forward pass
        z = inputs.float()
        for layer, activation in zip(self._linears, self._activations):
            z = activation(layer(z))

        if self._squeeze_out:
            out = torch.squeeze(z)
        else:
            out = z

        return out

    @staticmethod
    def activation_function(activation_name, params=None):
        """
        This functions returns the torch activation function.
        Args:
            activation_name (String): Name of the activation function.
            params (dict): Parameters for the activation function.

        """
        if activation_name == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation_name == 'tanh':
            return torch.nn.Tanh()
        elif activation_name == 'biased_tanh':
            return BiasedTanh(**params) if params is not None else BiasedTanh()
        elif activation_name == 'relu':
            return torch.nn.ReLU()
        elif activation_name == 'leaky_relu':
            return torch.nn.LeakyReLU(**params) if params is not None else torch.nn.LeakyReLU()
        elif activation_name == 'selu':
            return torch.nn.SELU()
        elif activation_name == 'identity':
            return torch.nn.Identity()
        elif activation_name == 'softplus':
            return torch.nn.Softplus()
        elif activation_name == 'softplustransformed':
            return SoftPlusTransformed(**params)
        elif activation_name == 'softplustransformed_indv':
            return SoftPlusTransformedIndividual(**params)
        elif activation_name == 'scaledsigmoid':
            return ScaledSigmoid(**params)
        else:
            raise ValueError('The activation %s in not supported.' % activation_name)


class NormcInitializer:

    def __init__(self, std=1.0):
        self._std = std

    def __call__(self, tensor):
        with torch.no_grad():
            tensor.normal_(std=self._std)
            tensor /= torch.sqrt(torch.sum(torch.square(tensor)))
            return tensor


class XavierInitializer:

    def __init__(self, gain):
        assert gain is not None
        self._gain = gain

    def __call__(self, tensor):
        torch.nn.init.xavier_uniform_(tensor, gain=self._gain)


class Standardizer(nn.Module):

    def __init__(self, use_cuda=False):
        # call base constructor
        super(Standardizer, self).__init__()

        self._sum = 0.0
        self._sumsq = 1e-2
        self._count = 1e-2
        self._use_cuda = use_cuda

        self.mean = 0.0
        self.std = 1.0

    def forward(self, inputs):
        self.update_mean_std(inputs.detach().cpu().numpy())
        mean = torch.tensor(self.mean).cuda() if self._use_cuda else torch.tensor(self.mean)
        std = torch.tensor(self.std).cuda() if self._use_cuda else torch.tensor(self.std)
        return (inputs - mean) / std

    def update_mean_std(self, x):
        self._sum += x.sum(axis=0).ravel()
        self._sumsq += np.square(x).sum(axis=0).ravel()
        self._count += np.array([len(x)])
        self.mean = self._sum / self._count
        self.std = np.sqrt(np.maximum((self._sumsq / self._count) - np.square(self.mean), 1e-2))



class BiasedTanh(torch.nn.Module):

    def __init__(self, mult=0.5, bias=0.5):
        super(BiasedTanh, self).__init__()
        self._bias = bias
        self._mult = mult

    def forward(self, input):
        return self._mult * torch.tanh(input) + self._bias


class SoftPlusTransformed(torch.nn.Module):

    def __init__(self, intercept_ordinate=0.6931, bias=0.0, threshold=20):
        super(SoftPlusTransformed, self).__init__()
        assert intercept_ordinate > bias, "The ordinate intercept is not allowed to be smaller" \
                                          "than or equal to the bias!"
        self._beta = np.log(2) / (intercept_ordinate - bias)
        self._bias = bias
        self._threshold = threshold
        self._softplus = torch.nn.Softplus(self._beta, self._threshold)

    def forward(self, input):
        out = self._softplus(input) + torch.ones_like(input) * self._bias
        return out


class SoftPlusTransformedIndividual(torch.nn.Module):

    def __init__(self, intercept_ordinate, bias, threshold=20):
        super().__init__()
        assert len(intercept_ordinate) == len(bias)
        for i, b in zip(intercept_ordinate, bias):
            assert i > b, "The ordinate intercept is not allowed to be smaller" \
                          "than or equal to the bias!"
        intercept_ordinate = np.array(intercept_ordinate)
        bias = np.array(bias)
        self._beta = np.log(2) / (intercept_ordinate - bias)
        self._bias = bias
        self._threshold = threshold
        self._softpluses = ModuleList([torch.nn.Softplus(beta=self._beta[i], threshold=threshold)
                                       for i in range(len(self._beta))])

    def forward(self, input):
        input = torch.atleast_2d(input)
        inner_dim = input.shape[-1]
        assert len(input.shape) == 2
        assert inner_dim == len(self._bias)
        out = torch.empty_like(input)
        for i in range(inner_dim):
            out[:, i] = self._softpluses[i](input[:, i]) + torch.ones_like(input[:, i]) * self._bias[i]
        return out


class ScaledSigmoid(torch.nn.Module):

    def __init__(self, mult):
        super().__init__()
        self._mult = mult

    def forward(self, input):
        scaled_input = input / self._mult
        return torch.sigmoid(scaled_input) * self._mult


class PPONetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(PPONetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        self._activation = torch.nn.ReLU()

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu')*0.01)
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu')*0.01)
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear')*0.01)

    def forward(self, state, **kwargs):
        features1 = self._activation(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._activation(self._h2(features1))
        a = self._h3(features2)

        return a


class PPONetwork_prev_a(PPONetwork):

    def forward(self, state, prev_action, **kwargs):
        inputs = torch.concat([state, prev_action], dim=1)
        features1 = self._activation(self._h1(torch.squeeze(inputs, 1).float()))
        features2 = self._activation(self._h2(features1))
        a = self._h3(features2)

        return a


class PPOWindowNetwork(PPONetwork):

    def __init__(self, input_shape, output_shape, dim_env_state, **kwargs):
        self._dim_env_state = dim_env_state
        super(PPOWindowNetwork, self).__init__(input_shape, output_shape, **kwargs)

    def forward(self, state, prev_action, lengths, **kwargs):
        env_state, _ = self.divide_state_to_env_hidden_batch_seq(state)

        # concatenate all states
        env_state = torch.flatten(env_state, start_dim=1)

        return super(PPOWindowNetwork, self).forward(env_state)

    def divide_state_to_env_hidden_batch_seq(self, states):
        assert len(states.shape) > 2, "This function only divides batches of sequences of states."
        return states[:, :, 0:self._dim_env_state], states[:, :,  self._dim_env_state:]


class TD3CriticBaseNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        dim_action = kwargs['action_shape'][0]
        dim_state = n_input - dim_action
        n_output = output_shape[0]

        # Assume there are two hidden layers
        assert len(n_features) == 2, 'TD3 critic needs 2 hidden layers'

        self._h1 = nn.Linear(dim_state + dim_action, n_features[0])
        self._h2_s = nn.Linear(n_features[0], n_features[1])
        self._h2_a = nn.Linear(dim_action, n_features[1], bias=False)
        self._h3 = nn.Linear(n_features[1], n_output)
        self._act_func = torch.nn.ReLU()

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2_s, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_s.weight)
        nn.init.uniform_(self._h2_s.weight, a=-1 / np.sqrt(fan_in_h2_s), b=1 / np.sqrt(fan_in_h2_s))

        fan_in_h2_a, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_a.weight)
        nn.init.uniform_(self._h2_a.weight, a=-1 / np.sqrt(fan_in_h2_a), b=1 / np.sqrt(fan_in_h2_a))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state, action):
        state = state.float()
        action = action.float()
        state_action = torch.cat((state, action), dim=1)

        features1 = self._act_func(self._h1(state_action))
        features2_s = self._h2_s(features1)
        features2_a = self._h2_a(action)
        features2 = self._act_func(features2_s + features2_a)

        q = self._h3(features2)
        return torch.squeeze(q)


class TD3ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, squash_and_scale_action=True, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = output_shape[0]

        self._action_scaling = torch.tensor(kwargs['action_scaling']).to(
            device=torch.device('cuda' if kwargs['use_cuda'] else 'cpu'))

        # Assume there are two hidden layers
        assert len(n_features) == 2, 'DDPG critic needs two hidden layers'

        self._h1 = nn.Linear(dim_state, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], dim_action)
        self._act_func = torch.nn.ReLU()
        self._squash_and_scale_action = squash_and_scale_action

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2, _ = nn.init._calculate_fan_in_and_fan_out(self._h2.weight)
        nn.init.uniform_(self._h2.weight, a=-1 / np.sqrt(fan_in_h2), b=1 / np.sqrt(fan_in_h2))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state):
        state = state.float()

        features1 = self._act_func(self._h1(state))
        features2 = self._act_func(self._h2(features1))
        a = self._h3(features2)

        if self._squash_and_scale_action:
            a = self._action_scaling * torch.tanh(a)

        return a


class SACCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        self._activation = torch.nn.ReLU()

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = self._activation(self._h1(state_action))
        features2 = self._activation(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class SACActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SACActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        self._activation = torch.nn.ReLU()

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear')*0.01)

    def forward(self, state):
        features1 = self._activation(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._activation(self._h2(features1))
        a = self._h3(features2)

        return a
