from enum import Enum
import math
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import torchdiffeq as tdeq
from s2pg.networks.vanilla_networks import ScaledSigmoid
from mushroom_rl.utils.torch import to_float_tensor


class ParameterTransformMode(Enum):
    C_SQUARED = 0
    C_EXP = 1
    B_SQUASHED = 2


class CPGActor(nn.Module):
    """
    Implementation of a Central Pattern Generator
    that integrates some form of ordinary differential equation
    to produce oscillating action output.
    """

    def __init__(self, input_shape, output_shape, oscillator_ode, ode_solver="rk4",
                 use_normalized_cpg_state=False, **kwargs):
        """
        Constructor of the CPGActor.
        :param oscillator_ode: The oscillator ode to be integrated for the action output
                               See library odeint for further information
        """
        super(CPGActor, self).__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._ode_solver = ode_solver
        self.oscillator_ode = oscillator_ode
        self.dt = self.oscillator_ode.dt
        self.max_psi = 0.0
        self.max_r = 0.0
        self.max_r_d = 0.0
        self._horizon = 1000
        self.low, self.high = self.get_low_high_cpg_state()
        self.norm_act_mean = torch.tensor((self.high + self.low) / 2.0)
        self.norm_act_delta = torch.tensor((self.high - self.low) / 2.0)
        self.use_normalized_cpg_state = use_normalized_cpg_state
        if use_normalized_cpg_state:
            self.low = -np.ones_like(self.low)
            self.high = np.ones_like(self.high)

    def forward(self, states):
        """
        Public forward function of this module.
        :param states: Tensor of cpg states.
        :return: Returns the next cpg state.
        """
        states = states.float()

        if self.use_normalized_cpg_state:
            # unnormalize state
            states = (states[:] * self.norm_act_delta) + self.norm_act_mean

        time_step = torch.tensor([0.0, self.dt], dtype=torch.float32)
        next_state = odeint(self.oscillator_ode, states, time_step, method=self._ode_solver)[-1]
        self.update_max_state_tracker(next_state)

        if self.use_normalized_cpg_state:
            # normalize states
            next_state = (next_state[:] - self.norm_act_mean) / self.norm_act_delta

        return next_state

    def update_max_state_tracker(self, next_state):
        max_psi = torch.max(next_state[:, self.oscillator_ode.psi_mask]).detach().cpu().numpy()
        max_r = torch.max(next_state[:, self.oscillator_ode.r_mask]).detach().cpu().numpy()
        max_r_d = torch.max(next_state[:, self.oscillator_ode.rd_mask]).detach().cpu().numpy()
        self.max_psi = max_psi if max_psi > self.max_psi else self.max_psi
        self.max_r = max_r if max_r > self.max_r else self.max_r
        self.max_r_d = max_r_d if max_r_d > self.max_r_d else self.max_r_d

    def get_state_dependent_lows_highs(self, states):
        states = to_float_tensor(states)
        states = torch.atleast_2d(states)
        time_step = torch.tensor([0.0, self.dt], dtype=torch.float32)
        lows = odeint(self.oscillator_ode.get_current_low_derivatives, states, time_step, method=self._ode_solver)[-1]
        highs = odeint(self.oscillator_ode.get_current_high_derivatives, states, time_step, method=self._ode_solver)[-1]
        return lows.detach().cpu().numpy(), highs.detach().cpu().numpy()

    def get_low_high_cpg_state(self):

        # r min is given by 0 and r max is given by b_max
        r_min = 0.0
        r_max = self.oscillator_ode.b_max

        # psi_min is 0 and psi_max can be calculated
        max_coupling = self.oscillator_ode.w_max * self.oscillator_ode.b_max * self.oscillator_ode.n_oscillators
        psi_min = -max_coupling    # this is only the case if v_min == 0!
        psi_max = self._horizon * self.dt * 2 * np.pi * self.oscillator_ode.v_max + max_coupling

        # r_d min and max are more complicated... We can calculate by running the ode with for r with the maximum params
        # todo: we can also do this in closed form instead of numerically!
        def r_ode(t, R):
            r = R[0]
            rd = R[1]
            rdd = self.oscillator_ode.c_max * ((self.oscillator_ode.c_max / 4) * (self.oscillator_ode.b_max - r) - rd)
            return torch.concat([torch.unsqueeze(rd, dim=0), torch.unsqueeze(rdd, dim=0)])

        t = torch.tensor(np.arange(0, self._horizon * self.dt, self.dt))
        r_rd = tdeq.odeint(r_ode, torch.tensor([0.0, 0.0]), t, method=self._ode_solver)
        rd_max = torch.max(r_rd[:, 1]).detach().cpu().numpy()
        rd_min = -rd_max
        n_oscillators = self.oscillator_ode.n_oscillators
        #low_psi = torch.tile(torch.tensor([psi_min]), dims=(n_oscillators,)).detach().cpu().numpy()
        #high_psi = torch.tile(torch.tensor([psi_max, r_max, rd_max]), dims=(n_oscillators,)).detach().cpu().numpy()
        #low_r = torch.tile(torch.tensor([r_min]), dims=(n_oscillators,)).detach().cpu().numpy()
        #high_r = torch.tile(torch.tensor([r_max]), dims=(n_oscillators,)).detach().cpu().numpy()

        low_psi = [psi_min for i in range(n_oscillators)]
        high_psi = [psi_max for i in range(n_oscillators)]
        low_r = [r_min for i in range(n_oscillators)]
        high_r = [r_max for i in range(n_oscillators)]
        low_rd = [rd_min for i in range(n_oscillators)]
        high_rd = [rd_max for i in range(n_oscillators)]

        low = np.concatenate([low_psi, low_r, low_rd])
        high = np.concatenate([high_psi, high_r, high_rd])
        return (low, high)


class CPGActorState(CPGActor):

    def __init__(self, input_shape, output_shape, dim_env_state, **kwargs):
        super().__init__(input_shape, output_shape, **kwargs)

        self._dim_env_state = dim_env_state

    def forward(self, states):
        """
        Public forward function of this module.
        :param states: Tensor of cpg states.
        :return: Returns the next cpg state.
        """
        states = states.float()
        time_step = torch.tensor([0.0, self.dt], dtype=torch.float32)
        next_state = odeint(self.oscillator_ode, states, time_step, method=self._ode_solver)[-1]
        self.update_max_state_tracker(next_state)
        return next_state[:, self._dim_env_state:]

class DeterministicCPGActor(CPGActor):

    def forward_full_trajectory(self, start_state, n_steps):
        start_state = start_state.float()
        time_steps = torch.tensor([i*self.dt for i in range(n_steps)], dtype=torch.float32) #TODO: Correct?
        trajectory_states = tdeq.odeint_adjoint.odeint(self.oscillator_ode, start_state, time_steps, method='rk4') #TODO: Would this make the other cases faster?
        return trajectory_states


class HopfOscillatorModule(nn.Module):

    DIM_CPG_STATE = 3

    """

    """
    def __init__(self, n_oscillators, dt, parameter_transform_modes=None, parameter_init=None,
                 fixed_parameters=None, use_cuda=False):
        """

        :param n_oscillators:
        :param dt:
        :param parameter_transform_modes:
        :param parameter_init:
        :param fixed_parameters:
        """
        super(HopfOscillatorModule, self).__init__()

        self.parameter_transform_modes = {} if parameter_transform_modes is None else parameter_transform_modes
        self.parameter_init = {'v': (0.05, 0.01), 'b': (0.1, 0.1), 'c': (2.0, 0.1),
                               'w': (0.1, 0.05), 'phi': (0.1, 0.05)} if parameter_init is None else parameter_init
        self.fixed_parameters = [] if fixed_parameters is None else fixed_parameters


        assert 'v' in self.parameter_init.keys()
        assert len(self.parameter_init['v']) == 2
        assert 'b' in self.parameter_init.keys()
        assert len(self.parameter_init['b']) == 2
        assert 'c' in self.parameter_init.keys()
        assert len(self.parameter_init['c']) == 2
        assert 'w' in self.parameter_init.keys()
        assert len(self.parameter_init['w']) == 2
        assert 'phi' in self.parameter_init.keys()
        assert len(self.parameter_init['phi']) == 2

        self.dt = dt
        self.n_oscillators = n_oscillators
        self._use_cuda = use_cuda


        if 'v' not in self.fixed_parameters:
            self.v = nn.Parameter(torch.empty(self.n_oscillators), requires_grad=True)
        else:
            self.v = torch.empty(self.n_oscillators)

        if 'b' not in self.fixed_parameters:
            self.b = nn.Parameter(torch.empty(self.n_oscillators), requires_grad=True)
        else:
            self.b = torch.empty(self.n_oscillators)

        if 'c' not in self.fixed_parameters:
            self.c = nn.Parameter(torch.empty(self.n_oscillators), requires_grad=True)
        else:
            self.c = torch.empty(self.n_oscillators)

        if 'w' not in self.fixed_parameters:
            self.w = nn.Parameter(torch.empty((self.n_oscillators-1, self.n_oscillators)), requires_grad=True)
        else:
            self.w = torch.empty((self.n_oscillators-1, self.n_oscillators))

        if 'phi' not in self.fixed_parameters:
            self.phi = nn.Parameter(torch.empty((self.n_oscillators-1, self.n_oscillators)), requires_grad=True)
        else:
            self.phi = torch.empty((self.n_oscillators-1, self.n_oscillators))

        self.psi_mask = np.arange(0, n_oscillators)
        self.r_mask = np.arange(n_oscillators, 2*n_oscillators)
        self.rd_mask = np.arange(2*n_oscillators, 3*n_oscillators)

        nn.init.normal_(self.v, mean=self.parameter_init['v'][0], std=self.parameter_init['v'][1])
        nn.init.normal_(self.b, mean=self.parameter_init['b'][0], std=self.parameter_init['b'][1])
        nn.init.normal_(self.c, mean=self.parameter_init['c'][0], std=self.parameter_init['c'][1])
        nn.init.normal_(self.w, mean=self.parameter_init['w'][0], std=self.parameter_init['w'][1])
        nn.init.normal_(self.phi, mean=self.parameter_init['phi'][0], std=self.parameter_init['phi'][1])

        self.v_max = 5.0
        self.v_min = 0.0
        self.b_max = 2.0
        self.b_min = 0.0
        self.c_max = 10.0
        self.c_min = 0.0
        self.w_max = 1.0
        self.w_min = 0.0
        self.phi_max = 2*np.pi
        self.phi_min = 0.0

        self.v_activation = ScaledSigmoid(self.v_max)
        self.b_activation = ScaledSigmoid(self.b_max)
        self.c_activation = ScaledSigmoid(self.c_max)
        self.w_activation = ScaledSigmoid(self.w_max)
        self.phi_activation = ScaledSigmoid(self.phi_max)

    def init_v_step_features(self, min_v, max_v):
        assert min_v < max_v
        v_vec = np.linspace(min_v, max_v, num=self.n_oscillators)
        self.v = nn.Parameter(torch.tensor(v_vec[:self.n_oscillators], dtype=torch.float32), requires_grad=True)

    def forward(self, t, cpg_states):

        n_cpg_states, psi, r, r_d = self.parse_cpg_state(cpg_states)

        v = self.v_activation(self.v)
        b = self.b_activation(self.b)
        c = self.c_activation(self.c)
        w = self.w_activation(self.w)
        phi = self.phi_activation(self.phi)

        return self.calculate_derivatives(psi, r, r_d, n_cpg_states, v, b, c, w, phi)

    def calculate_derivatives(self, psi, r, r_d, n_cpg_states, v, b, c, w, phi):

        # get weight matrix and phi
        w_zero_diag = self._get_zero_diag_mat_from_param(w)
        phi_zero_diag = self._get_zero_diag_mat_from_param(phi)

        # tile params to number of cpg_states
        c_eff = torch.tile(c, (n_cpg_states, 1, 1))
        b_eff = torch.tile(b, (n_cpg_states, 1, 1))
        w_zero_diag = torch.tile(w_zero_diag, (n_cpg_states, 1, 1))
        phi_zero_diag = torch.tile(phi_zero_diag, (n_cpg_states, 1, 1))
        v = torch.tile(v, (n_cpg_states, 1, 1))

        # Update r
        r_d_dot = c_eff * ((c_eff * 0.25) * (b_eff - r) - r_d)

        # Update psi
        psi_dot = 2 * torch.pi * v

        weight_mat = r * w_zero_diag
        Psi = torch.tile(psi, (self.n_oscillators, 1))
        Psi_T = torch.transpose(Psi, 1, 2)
        phase_mat = torch.sin(Psi - Psi_T - phi_zero_diag)
        coupling_term = (weight_mat * phase_mat).sum(dim=2)
        coupling_term = torch.unsqueeze(coupling_term, 1)
        psi_dot = psi_dot + coupling_term

        d_cpg_states = torch.cat([torch.squeeze(psi_dot, 1), torch.squeeze(r_d, 1), torch.squeeze(r_d_dot, 1)], dim=1).type(torch.float32)

        return d_cpg_states

    def get_current_high_derivatives(self, t, cpg_states):

        n_cpg_states, psi, r, r_d = self.parse_cpg_state(cpg_states)

        v = torch.ones_like(self.v) * self.v_max
        b = torch.ones_like(self.b) * self.b_max
        c = torch.ones_like(self.c) * self.c_max
        w = torch.ones_like(self.w) * self.w_max
        phi = torch.ones_like(self.phi) * self.phi_max

        return self.calculate_derivatives(psi, r, r_d, n_cpg_states, v, b, c, w, phi)

    def get_current_low_derivatives(self, t, cpg_states):

        n_cpg_states, psi, r, r_d = self.parse_cpg_state(cpg_states)

        v = torch.ones_like(self.v) * self.v_min
        b = torch.ones_like(self.b) * self.b_min
        c = torch.ones_like(self.c) * self.c_min
        w = torch.ones_like(self.w) * self.w_min
        phi = torch.ones_like(self.phi) * self.phi_min

        return self.calculate_derivatives(psi, r, r_d, n_cpg_states, v, b, c, w, phi)

    def parse_cpg_state(self, cpg_states):

        n_cpg_states = cpg_states.shape[0]
        psi = torch.unsqueeze(cpg_states[:, self.psi_mask], 1)
        r = torch.unsqueeze(cpg_states[:, self.r_mask], 1)
        r_d = torch.unsqueeze(cpg_states[:, self.rd_mask], 1)

        return n_cpg_states, psi, r, r_d

    def _get_zero_diag_mat_from_param(self, param):
        """

        :param param:
        :return:
        """
        zeros = torch.zeros((self.n_oscillators - 1, 1)).cuda()\
            if self._use_cuda else torch.zeros((self.n_oscillators - 1, 1))
        stacked = torch.hstack([zeros, param])
        zero = torch.zeros(1).cuda() if self._use_cuda else torch.zeros(1)
        res = torch.hstack([stacked.reshape(-1), zero]).reshape(self.n_oscillators, self.n_oscillators)
        return res

    def _get_transformed_c(self):
        if 'c' in self.parameter_transform_modes.keys():
            if ParameterTransformMode.C_EXP == self.parameter_transform_modes['c']:
                return self.c.exp()
            elif ParameterTransformMode.C_SQUARED == self.parameter_transform_modes['c']:
                return self.c ** 2
            else:
                return self.c
        else:
            return self.c

    def _get_transformed_b(self):
        if 'b' in self.parameter_transform_modes.keys():
            if ParameterTransformMode.B_SQUASHED == self.parameter_transform_modes['b']:
                return torch.tanh(self.b)
            else:
                return self.b
        else:
            return self.b

    def get_interpretable_w(self):
        return self.w.cpu().detach().numpy().reshape((self.n_oscillators, self.n_oscillators - 1))

    def get_interpretable_phi(self):
        return self.phi.cpu().detach().numpy().reshape((self.n_oscillators, self.n_oscillators - 1))


class HopfOscillatorModuleEvnState(nn.Module):

    def __init__(self, n_oscillators, dt, parameter_network, parameter_network_params, dim_env_state, sw=None, logging_iter=1000):

        # Todo: For now we fix the parameter transform modes. Also initialization is current disabled and w and phi are neglected.
        # call grand-parent constructor
        super().__init__()

        self.dt = dt
        self.n_oscillators = n_oscillators
        self._dim_env_state = dim_env_state

        self.psi_mask = np.arange(0, n_oscillators)
        self.r_mask = np.arange(n_oscillators, 2 * n_oscillators)
        self.rd_mask = np.arange(2 * n_oscillators, 3 * n_oscillators)
        self._parameter_network = parameter_network(**parameter_network_params)
        self._iter = 0
        self._logging_iter = logging_iter

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None


    def forward(self, t, cpg_states, vbc, dt):
        ind = t * (1.0/dt)
        ind = ind.int()

        curr_vbc= vbc[:, ind, :]
        curr_vbc = torch.atleast_2d(curr_vbc)

        n_cpg_states = cpg_states.shape[0]
        psi = torch.unsqueeze(cpg_states[:, self.psi_mask], 1)
        r = torch.unsqueeze(cpg_states[:, self.r_mask], 1)
        r_d = torch.unsqueeze(cpg_states[:, self.rd_mask], 1)

        # get the parameters
        v = curr_vbc[:, 0:self.n_oscillators].view(-1, 1, self.n_oscillators)
        b = curr_vbc[:, self.n_oscillators:2*self.n_oscillators].view(-1, 1, self.n_oscillators)
        c = curr_vbc[:, 2*self.n_oscillators:].view(-1, 1, self.n_oscillators)

        if self._sw and self._iter % self._logging_iter == 0:
            # todo: for rk4, logging is done 4 times
            for i in range(self.n_oscillators):
                self._sw.add_scalar("CPG %d Params/v" % i, torch.mean(v[:, 0, i]), self._iter)
                self._sw.add_scalar("CPG %d Params/b" % i, torch.mean(b[:, 0, i]), self._iter)
                self._sw.add_scalar("CPG %d Params/c" % i, torch.mean(c[:, 0, i]), self._iter)

        # get weight matrix and phi Todo: Currently disabled!
        #w_zero_diag = torch.zeros((self.n_oscillators, self.n_oscillators))
        #phi_zero_diag = torch.zeros((self.n_oscillators, self.n_oscillators))

        # tile params to number of cpg_states
        #w_zero_diag = torch.tile(w_zero_diag, (n_cpg_states, 1, 1))
        #phi_zero_diag = torch.tile(phi_zero_diag, (n_cpg_states, 1, 1))

        # Update r
        r_d_dot = c * ((c * 0.25) * (b - r) - r_d)

        # Update psi
        psi_dot = 2 * math.pi * v

        #weight_mat = r * w_zero_diag
        #Psi = torch.tile(psi, (self.n_oscillators, 1))
        #Psi_T = torch.transpose(Psi, 1, 2)
        #phase_mat = torch.sin(Psi - Psi_T - phi_zero_diag)
        #coupling_term = (weight_mat * phase_mat).sum(dim=2)
        #coupling_term = torch.unsqueeze(coupling_term, 1)
        #psi_dot = psi_dot + coupling_term

        d_cpg_states = torch.cat([torch.squeeze(psi_dot, 1),
                                  torch.squeeze(r_d, 1), torch.squeeze(r_d_dot, 1)], dim=1).type(torch.float32)

        self._iter += 1 # todo: for rk4, iter is called 4 times

        return d_cpg_states
