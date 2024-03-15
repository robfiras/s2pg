import numpy as np
import torch
from mushroom_rl.utils.torch import zero_grad
from mushroom_rl.approximators.parametric import TorchApproximator


class StatefulTorchApproximator(TorchApproximator):

    def diff_batch(self, states, prev_action, lengths, reduction_weights):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.

        Args:
            states (np.ndarray): the state;
            prev_action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """

        reduction_weights = torch.from_numpy(reduction_weights).type(torch.FloatTensor)

        batch_size = states.shape[0]
        if len(states.shape) == 3:
            # we are using bptt_pI, so we need sequences
            assert lengths is not None
            assert len(prev_action.shape) == 3
        else:
            # we are not using bptt_pI, but we extend the dims to use the same bptt_pI network with sequence lengths 1
            states = states.reshape(batch_size, 1, -1)
            prev_action = prev_action.reshape(batch_size, 1, -1)
            lengths = np.ones(batch_size, dtype=int)

        if not self._use_cuda:
            states = torch.from_numpy(states)
            prev_action = torch.from_numpy(prev_action)
            lengths = torch.from_numpy(lengths)

        else:
            states = torch.from_numpy(states).cuda()
            prev_action = torch.from_numpy(prev_action).cuda()
            lengths = torch.from_numpy(lengths).cuda()

        y_hat = self.network(states, prev_action, lengths)

        assert len(reduction_weights) == len(y_hat)
        assert len(reduction_weights.shape) == 3
        y_hat = torch.unsqueeze(y_hat, 1)
        y_hat = torch.matmul(y_hat, reduction_weights)
        y_hat = torch.sum(y_hat)

        zero_grad(self.network.parameters())
        y_hat.backward(retain_graph=True)

        gradient = list()
        for p in self.network.parameters():
            g = p.grad.data.detach().cpu().numpy()
            gradient.append(g.flatten())

        g = np.concatenate(gradient, 0)

        return g

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        if not self._use_cuda:
            torch_args = [torch.from_numpy(np.atleast_2d(x)) for x in args]
        else:
            torch_args = [torch.from_numpy(np.atleast_2d(x)).cuda()
                          for x in args]

        y_hat = self.network(*torch_args, **kwargs)
        n_outs = 1 if len(y_hat.shape) == 0 else y_hat.shape[-1]
        y_hat = y_hat.view(-1, n_outs)

        gradients = list()
        for i in range(y_hat.shape[1]):
            zero_grad(self.network.parameters())
            y_hat[:, i].backward(retain_graph=True)

            gradient = list()
            for p in self.network.parameters():
                g = p.grad.data.detach().cpu().numpy()
                gradient.append(g.flatten())

            g = np.concatenate(gradient, 0)

            gradients.append(g)

        g = np.stack(gradients, -1)

        return g

    def get_weights_and_names(self):

        # Todo: This is not save to be used with weight matrices of linear layers, check this!

        weights = list()
        weight_names = list()

        for n, p in self.network.named_parameters():
            w = p.data.detach().cpu().numpy()
            weights.append(w.flatten())
            weight_names.append(n)

        weights = np.concatenate(weights, 0)

        return weights, weight_names

    def get_weight_names(self):

        weight_names = list()
        for n, p in self.network.named_parameters():
            for i in range(len(p.data.flatten().detach().cpu().numpy())):
                weight_names.append(n + "_%d" % i)

        return weight_names


class DeterministicTrajectoryTorchApproximator(TorchApproximator): #TODO: Which does this inherit?

    def diff_batch(self, states, reduction_weights):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        if not self._use_cuda:
            states = torch.from_numpy(np.atleast_2d(states))
        else:
            states = torch.from_numpy(np.atleast_2d(states)).cuda()

        reduction_weights = torch.from_numpy(reduction_weights)

        y_hat = self.network(states)

        assert len(reduction_weights) == len(y_hat)
        assert len(reduction_weights.shape) == 3
        y_hat = torch.unsqueeze(y_hat, 1)
        y_hat = torch.matmul(y_hat, reduction_weights)
        y_hat = torch.sum(y_hat)

        zero_grad(self.network.parameters())
        y_hat.backward(retain_graph=True)

        gradient = list()
        for p in self.network.parameters():
            g = p.grad.data.detach().cpu().numpy()
            gradient.append(g.flatten())

        g = np.concatenate(gradient, 0)

        return g
