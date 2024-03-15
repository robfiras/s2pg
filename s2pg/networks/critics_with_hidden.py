import torch
from .vanilla_networks import FullyConnectedNetwork, XavierInitializer


class CriticHiddenState(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim

        self._input_encoder = FullyConnectedNetwork(input_shape=input_shape, output_shape=(100,), n_features=[512, 256],
                                                    activations=["relu", "relu", "tanh"])
        self._action_encoder = FullyConnectedNetwork(input_shape=(action_dim,), output_shape=(100,), n_features=[512],
                                                     activations=["relu", "tanh"])
        self._joint_encoder = FullyConnectedNetwork(input_shape=(200,), output_shape=output_shape, n_features=[256],
                                                    activations=["relu", "identity"])

    def forward(self, state_hidden_state, action, next_hidden_state):
        encoding_1 = self._input_encoder(state_hidden_state, action, next_hidden_state)
        encoding_2 = self._action_encoder(action)
        q = self._joint_encoder(encoding_1, encoding_2)
        return torch.squeeze(q)


class CriticHiddenStatev2(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, dim_env_state, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state

        self._hidden_encoder = FullyConnectedNetwork(input_shape=(2 * self._hidden_state_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "tanh"])
        self._state_action_encoder = FullyConnectedNetwork(input_shape=(state_dim + action_dim,),
                                                           output_shape=(256,),
                                                           n_features=[512],
                                                           activations=["relu", "tanh"])
        self._action_encoder = FullyConnectedNetwork(input_shape=(action_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "tanh"])
        self._joint_encoder = FullyConnectedNetwork(input_shape=(256 * 3,),
                                                    output_shape=output_shape,
                                                    n_features=[256],
                                                    activations=["relu", "identity"])

    def forward(self, state_hidden_state, action, next_hidden_state):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]
        encoding_1 = self._hidden_encoder(hidden_state, next_hidden_state)
        encoding_2 = self._state_action_encoder(env_state, action)
        encoding_3 = self._action_encoder(action)
        q = self._joint_encoder(encoding_1, encoding_2, encoding_3)
        return torch.squeeze(q)


class CriticHiddenStatev2_prev_action(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, dim_env_state, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state

        self._hidden_encoder = FullyConnectedNetwork(input_shape=(2 * self._hidden_state_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "tanh"])
        self._state_action_encoder = FullyConnectedNetwork(input_shape=(state_dim + 2 * action_dim,),
                                                           output_shape=(256,),
                                                           n_features=[512],
                                                           activations=["relu", "tanh"])
        self._action_encoder = FullyConnectedNetwork(input_shape=(2 * action_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "tanh"])
        self._joint_encoder = FullyConnectedNetwork(input_shape=(256 * 3,),
                                                    output_shape=output_shape,
                                                    n_features=[256],
                                                    activations=["relu", "identity"])

    def forward(self, state_hidden_state, action, next_hidden_state, prev_action):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]
        encoding_1 = self._hidden_encoder(hidden_state, next_hidden_state)
        encoding_2 = self._state_action_encoder(env_state, action, prev_action)
        encoding_3 = self._action_encoder(action, prev_action)
        q = self._joint_encoder(encoding_1, encoding_2, encoding_3)
        return torch.squeeze(q)


class CriticHiddenStatev3(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, dim_env_state, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state

        self._hidden_encoder = FullyConnectedNetwork(input_shape=(2 * self._hidden_state_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "identity"])
        self._state_action_encoder = FullyConnectedNetwork(input_shape=(state_dim + action_dim,),
                                                           output_shape=(256,),
                                                           n_features=[512],
                                                           activations=["relu", "identity"])
        self._action_encoder = FullyConnectedNetwork(input_shape=(action_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "identity"])
        self._joint_encoder = FullyConnectedNetwork(input_shape=(256,),
                                                    output_shape=output_shape,
                                                    n_features=[256],
                                                    activations=["relu", "identity"])

    def forward(self, state_hidden_state, action, next_hidden_state):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]
        encoding_1 = self._hidden_encoder(hidden_state, next_hidden_state)
        encoding_2 = self._state_action_encoder(env_state, action)
        encoding_3 = self._action_encoder(action)
        combined = torch.nn.functional.relu(encoding_1 + encoding_2 + encoding_3)
        q = self._joint_encoder(combined)
        return torch.squeeze(q)


class CriticHiddenStatev3_prev_action(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, dim_env_state,
                 use_prev_action=True, use_cuda=False, n_features=512, squeeze_out=True, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state
        self._use_prev_action = use_prev_action
        self._squeeze_out = squeeze_out

        def gain_calc(x): return torch.nn.init.calculate_gain(x) * 0.1

        self._hidden_encoder = FullyConnectedNetwork(input_shape=(2 * self._hidden_state_dim,),
                                                     output_shape=(n_features//2,),
                                                     n_features=[n_features],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))])

        input_shape_state_action_enc = state_dim+2*action_dim if self._use_prev_action else state_dim+action_dim
        self._state_action_encoder = FullyConnectedNetwork(input_shape=(input_shape_state_action_enc,),
                                                           output_shape=(n_features//2,),
                                                           n_features=[n_features],
                                                           activations=["relu", "identity"],
                                                           initializers=[XavierInitializer(gain_calc("relu")),
                                                                         XavierInitializer(gain_calc("linear"))]
                                                           )

        input_shape_action_enc = 2*action_dim if self._use_prev_action else action_dim
        self._action_encoder = FullyConnectedNetwork(input_shape=(input_shape_action_enc,),
                                                     output_shape=(n_features//2,),
                                                     n_features=[n_features],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))]
                                                     )

        self._joint_encoder = FullyConnectedNetwork(input_shape=(n_features // 2,),
                                                    output_shape=output_shape,
                                                    n_features=[n_features//2],
                                                    activations=["relu", "identity"],
                                                    initializers=[XavierInitializer(gain_calc("relu")),
                                                                  XavierInitializer(gain_calc("linear"))]
                                                    )

    def forward(self, state_hidden_state, action, next_hidden_state, prev_action):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]

        # useful transformation for cpg
        #hidden_state = torch.concat([torch.sin(hidden_state), torch.cos(hidden_state)], dim=1)
        #next_hidden_state = torch.concat([torch.sin(next_hidden_state), torch.cos(next_hidden_state)], dim=1)
        #hidden_state = torch.arctanh(hidden_state * (1-1e-7))
        #next_hidden_state = torch.arctanh(next_hidden_state * (1-1e-7))

        encoding_1 = self._hidden_encoder(hidden_state, next_hidden_state)
        if self._use_prev_action:
            encoding_2 = self._state_action_encoder(env_state, action, prev_action)
            encoding_3 = self._action_encoder(action, prev_action)
        else:
            encoding_2 = self._state_action_encoder(env_state, action)
            encoding_3 = self._action_encoder(action)
        combined = torch.nn.functional.relu(encoding_1 + encoding_2 + encoding_3)
        q = self._joint_encoder(combined)

        if self._squeeze_out:
            return torch.squeeze(q)
        else:
            return q


class CriticHiddenStatev3_prev_action_single(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, dim_env_state,
                 use_prev_action=True, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state
        self._use_prev_action = use_prev_action

        def gain_calc(x): return torch.nn.init.calculate_gain(x) * 0.1

        self._hidden_encoder = FullyConnectedNetwork(input_shape=(self._hidden_state_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))])

        input_shape_state_action_enc = state_dim+2*action_dim if self._use_prev_action else state_dim+action_dim
        self._state_action_encoder = FullyConnectedNetwork(input_shape=(input_shape_state_action_enc,),
                                                           output_shape=(256,),
                                                           n_features=[512],
                                                           activations=["relu", "identity"],
                                                           initializers=[XavierInitializer(gain_calc("relu")),
                                                                         XavierInitializer(gain_calc("linear"))]
                                                           )

        input_shape_action_enc = 2*action_dim if self._use_prev_action else action_dim
        self._action_encoder = FullyConnectedNetwork(input_shape=(input_shape_action_enc,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))]
                                                     )
        self._joint_encoder = FullyConnectedNetwork(input_shape=(256,),
                                                    output_shape=output_shape,
                                                    n_features=[256],
                                                    activations=["relu", "identity"],
                                                    initializers=[XavierInitializer(gain_calc("relu")),
                                                                  XavierInitializer(gain_calc("linear"))]
                                                    )

    def forward(self, state_hidden_state, action, prev_action):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]

        # useful transformation for cpg
        #hidden_state = torch.concat([torch.sin(hidden_state), torch.cos(hidden_state)], dim=1)
        #next_hidden_state = torch.concat([torch.sin(next_hidden_state), torch.cos(next_hidden_state)], dim=1)
        #hidden_state = torch.arctanh(hidden_state * (1-1e-7))
        #next_hidden_state = torch.arctanh(next_hidden_state * (1-1e-7))

        encoding_1 = self._hidden_encoder(hidden_state)
        if self._use_prev_action:
            encoding_2 = self._state_action_encoder(env_state, action, prev_action)
            encoding_3 = self._action_encoder(action, prev_action)
        else:
            encoding_2 = self._state_action_encoder(env_state, action)
            encoding_3 = self._action_encoder(action)
        combined = torch.nn.functional.relu(encoding_1 + encoding_2 + encoding_3)
        q = self._joint_encoder(combined)
        return torch.squeeze(q)


class CriticHiddenStatev3_prev_action_singlev2(CriticHiddenStatev3_prev_action_single):

    def forward(self, env_state, action, hidden_state, prev_action):

        encoding_1 = self._hidden_encoder(hidden_state)
        encoding_2 = self._state_action_encoder(env_state, action, prev_action)
        encoding_3 = self._action_encoder(action, prev_action)
        combined = torch.nn.functional.relu(encoding_1 + encoding_2 + encoding_3)
        q = self._joint_encoder(combined)
        return torch.squeeze(q)


class CriticHiddenStatev4_prev_action(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim, dim_env_state,
                 use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._dim_env_state = dim_env_state

        self._network = FullyConnectedNetwork(input_shape=(self._state_dim + 2 * self._action_dim +
                                                           2 * self._hidden_state_dim,),
                                              output_shape=(1,),
                                              n_features=[512, 256],
                                              activations=["relu", "relu", "identity"])

    def forward(self, state_hidden_state, action, next_hidden_state, prev_action):
        env_state = state_hidden_state[:, 0:self._dim_env_state]
        hidden_state = state_hidden_state[:, self._dim_env_state:]
        hidden_state = torch.arctanh(hidden_state * (1-1e-7))
        next_hidden_state = torch.arctanh(next_hidden_state * (1-1e-7))
        q = self._network(env_state, hidden_state, action, next_hidden_state, prev_action)
        return torch.squeeze(q)


class CriticBPTT_Hidden(torch.nn.Module):

    def __init__(self, input_shape, output_shape, state_dim, action_dim, hidden_state_dim,
                 use_prev_action=True, use_cuda=False, **kwargs):

        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_state_dim = hidden_state_dim
        self._use_prev_action = use_prev_action

        def gain_calc(x): return torch.nn.init.calculate_gain(x) * 0.1

        self._hidden_encoder = FullyConnectedNetwork(input_shape=(self._hidden_state_dim,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))])

        input_shape_state_action_enc = state_dim+2*action_dim if self._use_prev_action else state_dim+action_dim
        self._state_action_encoder = FullyConnectedNetwork(input_shape=(input_shape_state_action_enc,),
                                                           output_shape=(256,),
                                                           n_features=[512],
                                                           activations=["relu", "identity"],
                                                           initializers=[XavierInitializer(gain_calc("relu")),
                                                                         XavierInitializer(gain_calc("linear"))]
                                                           )

        input_shape_action_enc = 2*action_dim if self._use_prev_action else action_dim
        self._action_encoder = FullyConnectedNetwork(input_shape=(input_shape_action_enc,),
                                                     output_shape=(256,),
                                                     n_features=[512],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))]
                                                     )
        self._joint_encoder1 = FullyConnectedNetwork(input_shape=(256,),
                                                     output_shape=(1,),
                                                     n_features=[256],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))]
                                                     )

        self._joint_encoder2 = FullyConnectedNetwork(input_shape=(256,),
                                                     output_shape=(1,),
                                                     n_features=[256],
                                                     activations=["relu", "identity"],
                                                     initializers=[XavierInitializer(gain_calc("relu")),
                                                                   XavierInitializer(gain_calc("linear"))]
                                                     )

    def forward(self, env_state, next_hidden_state, action, prev_action):
        encoding_1 = self._hidden_encoder(next_hidden_state)
        if self._use_prev_action:
            encoding_2 = self._state_action_encoder(env_state, action, prev_action)
            encoding_3 = self._action_encoder(action, prev_action)
        else:
            encoding_2 = self._state_action_encoder(env_state, action)
            encoding_3 = self._action_encoder(action)
        combined = torch.nn.functional.relu(encoding_1 + encoding_2 + encoding_3)
        q1 = self._joint_encoder1(combined)
        q2 = self._joint_encoder2(combined)
        return torch.concat([q1, q2], dim=1)


class CriticHiddenStatev5(FullyConnectedNetwork):

    def __init__(self, input_shape, output_shape, use_prev_action=False,  **kwargs):
        self._use_prev_action = use_prev_action
        super(CriticHiddenStatev5, self).__init__(input_shape, output_shape, **kwargs)

    def forward(self, state_hidden_state, action, next_hidden_state, prev_action):
        if self._use_prev_action:
            return super(CriticHiddenStatev5, self).forward(state_hidden_state, action, next_hidden_state, prev_action)
        else:
            return super(CriticHiddenStatev5, self).forward(state_hidden_state, action, next_hidden_state)


def get_hidden_critic(critic_type):
    if critic_type == "v1":
        return CriticHiddenState
    elif critic_type == "v2":
        return CriticHiddenStatev2
    elif critic_type == "v2_prev_action":
        return CriticHiddenStatev2_prev_action
    elif critic_type == "v3":
        return CriticHiddenStatev3
    elif critic_type == "v3_prev_action":
        return CriticHiddenStatev3_prev_action
    elif critic_type == "v3_prev_action_single":
        return CriticHiddenStatev3_prev_action_single
    elif critic_type == "v3_prev_action_singlev2":
        return CriticHiddenStatev3_prev_action_singlev2
    elif critic_type == "v4_prev_action":
        return CriticHiddenStatev4_prev_action
    elif critic_type == "v5":
        return CriticHiddenStatev5
    elif critic_type == "custom_fcn":
        return FullyConnectedNetwork
    elif critic_type == "bppt_our_critic":
        return CriticBPTT_Hidden
    else:
        raise ValueError("Unknown HiddenCritic version %s." % critic_type)
