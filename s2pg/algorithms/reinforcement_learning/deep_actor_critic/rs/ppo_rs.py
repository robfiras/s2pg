import warnings
from copy import  deepcopy
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


class PPO_RS(Agent):
    """
    Proximal Policy Optimization algorithm formulated with the stochastic stateful policy gradient.
    Implementation with a recurrent stochastic policy (hence RS) and a critic that takes the policy states
    into account. This can be used in settings with and without privileged information.

    Non-Recurrent version first appeared in:
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, hidden_state_dim,
                 ent_coeff=0.0, policy_state_mask=None, critic_state_mask=None,
                 new_state_update_tau=0.0, critic_fit_params=None, include_prev_a_in_V=False,
                 sw=None):
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

        self._dim_env_state = mdp_info.observation_space.shape[0]

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        super().__init__(mdp_info, policy, None)

        # remove preprocessor from save attributes
        del self._save_attributes["_preprocessors"]

        # add the standardization preprocessor
        self._standardizer = StandardizationPreprocessor_ext(mdp_info)
        self._preprocessors.append(self._standardizer)

        # add the preprocessor to append the hidden state to the environment state
        self._preprocessors.append(self.append_hidden_state)

        self._hidden_state_dim = hidden_state_dim
        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)
        self._critic_state_mask = critic_state_mask if critic_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0]+hidden_state_dim, dtype=bool)

        self._new_state_update_tau = new_state_update_tau
        self._include_prev_a_in_V = include_prev_a_in_V

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
            _standardizer='pickle',
            _dim_env_state='primitive',
            _policy_state_mask="numpy",
            _critic_state_mask="numpy",
            _new_state_update_tau="primitive",
            _include_prev_a_in_V='primitive',
        )

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def add_preprocessor(self, preprocessor):
        # for now disable the preprocessor to ensure that appending the hidden state is always done at last
        raise AttributeError("This agent current does not support preprocessors.")

    def add_hidden_state_preprocessor(self):
        if len(self._preprocessors) == 0:   # only the hidden state preprocessor is allowed for now, which is why we can check the length
            self._preprocessors.append(self.append_hidden_state)
        else:
            warnings.warn("Hidden state preprocessor already included, and will be not added twice.")

    def append_hidden_state(self, x):
        # get latest hidden state
        hidden_state = self.policy.get_last_hidden_state()
        return np.concatenate([x, hidden_state])

    def fit(self, dataset, **info):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        prev_u = self.get_prev_action(u, last, absorbing)

        obs, _ = self.divide_state_to_env_hidden_batch(x)
        _, next_hidden_obs = self.divide_state_to_env_hidden_batch(xn)
        next_hidden_obs = to_float_tensor(next_hidden_obs)
        obs_hidden_obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)
        prev_act = to_float_tensor(prev_u)

        v_target, np_adv = self.compute_gae(self._V, x, xn, r, prev_u, u, absorbing, last, self.mdp_info.gamma, self._lambda())
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(torch.unsqueeze(obs_hidden_obs[:, self._policy_state_mask], dim=1),
                                                  torch.unsqueeze(prev_act, dim=1))
        dist_input = torch.concat([act, next_hidden_obs], dim=1)
        old_log_p = old_pol_dist.log_prob(dist_input)[:, None].detach()

        if self._include_prev_a_in_V:
            self._V.fit(x[:, self._critic_state_mask], prev_u, v_target, **self._critic_fit_params)
        else:
            self._V.fit(x[:, self._critic_state_mask], prev_u, v_target, **self._critic_fit_params)

        self._update_policy(obs_hidden_obs, act, next_hidden_obs, prev_act, adv, old_log_p, last)

        # Print fit information
        self._log_info(dataset, x, prev_u, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, obs_hidden_obs, act, next_hidden_obs,  prev_act, adv, old_log_p, last):
        obs_hidden_obs = torch.unsqueeze(obs_hidden_obs[:, self._policy_state_mask], dim=1)
        prev_act = torch.unsqueeze(prev_act, dim=1)
        for epoch in range(self._n_epochs_policy()):
            for obs_hidden_obs_i, act_i, next_hidden_obs_i, prev_act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs_hidden_obs, act, next_hidden_obs, prev_act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_joint_t(obs_hidden_obs_i, prev_act_i, act_i, next_hidden_obs_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy_t(obs_hidden_obs_i)
                loss.backward()
                self._optimizer.step()

            # update hidden state in dataset
            if self._new_state_update_tau > 0.0:

                dist = self.policy.distribution_t(obs_hidden_obs, prev_act)
                out_new = dist.sample().detach()
                first_hidden_state = deepcopy(obs_hidden_obs[0, 0, -self._hidden_state_dim:]).view(1, -1)
                new_next_hidden = torch.squeeze(out_new)[:, -self._hidden_state_dim:]
                new_hidden = torch.concat([first_hidden_state, new_next_hidden[:-1]])
                curr_hidden = obs_hidden_obs[:, 0, -self._hidden_state_dim:]
                obs_hidden_obs[:, 0, -self._hidden_state_dim:] = (1 - self._new_state_update_tau) * curr_hidden \
                                                                  + self._new_state_update_tau * new_hidden

    def _log_info(self, dataset, x, prev_u, v_target, old_pol_dist):
        if self._sw:
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            if self._include_prev_a_in_V:
                v_pred = torch.tensor(self._V(x[:, self._critic_state_mask], prev_u), dtype=torch.float)
            else:
                v_pred = torch.tensor(self._V(x[:, self._critic_state_mask]), dtype=torch.float)
            v_err = F.mse_loss(v_pred, torch_v_targets)

            logging_ent = self.policy.entropy(x[:, self._policy_state_mask])
            x = np.expand_dims(x[:, self._policy_state_mask], axis=1)
            prev_u = np.expand_dims(prev_u, axis=1)
            new_pol_dist = self.policy.distribution(x, prev_u)
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(
                new_pol_dist, old_pol_dist))
            avg_rwd = np.mean(compute_J(dataset))
            try:
                L = int(np.round(np.mean(compute_episodes_length(dataset))))
            except:
                L = 0

            self._sw.add_scalar('Training Details/Training Reward', avg_rwd, self._iter)
            self._sw.add_scalar('Training Details/Mean Episode Length', L, self._iter)
            self._sw.add_scalar('Training Details/Value-Function Loss', v_err, self._iter)
            self._sw.add_scalar('Training Details/Mean Value', torch.mean(v_pred), self._iter)
            self._sw.add_scalar('Training Details/Entropy', logging_ent, self._iter)
            self._sw.add_scalar('Training Details/KL', logging_kl, self._iter)

            try:
                ode = self.policy._mu.model.network._cpg_function.oscillator_ode
                v = torch.mean(ode.v.detach())
                b = torch.mean(ode.b.detach())
                c = torch.mean(ode.c.detach())
                w = torch.mean(ode.w.detach())
                phi = torch.mean(ode.phi.detach())
                self._sw.add_scalar('CPG-Params/v', v, self._iter)
                self._sw.add_scalar('CPG-Params/b', b, self._iter)
                self._sw.add_scalar('CPG-Params/c', c, self._iter)
                self._sw.add_scalar('CPG-Params/w', w, self._iter)
                self._sw.add_scalar('CPG-Params/phi', phi, self._iter)

            except:
                pass

    def _post_load(self):
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))
        self._preprocessors = []
        self._preprocessors.append(self._standardizer)
        self._preprocessors.append(self.append_hidden_state)

    @staticmethod
    def get_prev_action(actions, lasts, absorbings):
        pa = list()
        for i in range(len(actions)):
            if i == 0 or lasts[i - 1] or absorbings[i - 1]:
                pa.append(np.zeros(len(actions[0])))
            else:
                pa.append(actions[i - 1])

        return np.array(pa)

    def compute_gae(self, V, s, ss, r, prev_u, u, absorbing, last, gamma, lam):
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
        if self._include_prev_a_in_V:
            v = V(s[:, self._critic_state_mask], prev_u)
            v_next = V(ss[:, self._critic_state_mask], u)
        else:
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

