import torch

from s2pg.networks.vanilla_networks import FullyConnectedNetwork


class DiscriminatorNetwork(FullyConnectedNetwork):
    def __init__(self, input_shape, output_shape, n_features, activations,  initializers=None,
                 squeeze_out=True, standardizer=None, use_actions=True, use_next_states=False, **kwargs):
        # call base constructor
        super(DiscriminatorNetwork, self).__init__(input_shape=input_shape, output_shape=output_shape, n_features=n_features,
                                                   activations=activations, initializers=initializers, squeeze_out=squeeze_out,
                                                   standardizer=standardizer, **kwargs)

        assert not (use_actions and use_next_states), "Discriminator with states, actions and next states as" \
                                                      "input currently not supported."

        self.use_actions = use_actions
        self.use_next_states = use_next_states

    def forward(self, *inputs):
        inputs = self.preprocess_inputs(*inputs)
        # define forward pass
        z = inputs.float()
        for layer, activation in zip(self._linears, self._activations):
            z = activation(layer(z))
        return z

    def preprocess_inputs(self, *inputs):
        if self.use_actions:
            states, actions = inputs
        elif self.use_next_states:
            states, next_states = inputs
        else:
            states = inputs[0]
        # normalize states
        if self._stand is not None:
            states = self._stand(states)
            if self.use_next_states:
                next_states = self._stand(next_states)
        if self.use_actions:
            inputs = torch.squeeze(torch.cat([states, actions], dim=1), 1)
        elif self.use_next_states:
            inputs = torch.squeeze(torch.cat([states, next_states], dim=1), 1)
        else:
            inputs = states
        return inputs
