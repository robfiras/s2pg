import numpy as np

import torch
import torch.nn.functional as F

from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor, update_optimizer_parameters
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import parse_dataset, compute_J, compute_episodes_length
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.utils.parameters import to_parameter


class PPO(Agent):
    """s
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam,  policy_state_mask=None,
                 ent_coeff=0.0, critic_fit_params=None, standardizer=None, sw=None):
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

        self._policy_state_mask = policy_state_mask if policy_state_mask is not None else\
            np.ones(mdp_info.observation_space.shape[0], dtype=bool)

        self._standardizer = standardizer

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

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
            _policy_state_mask='numpy'
        )

        super().__init__(mdp_info, policy, None)

    def fit(self, dataset, **info):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        # update running mean and std
        if self._standardizer:
            self._standardizer.update_mean_std(x)

        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda())
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(obs[:, self._policy_state_mask])
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x, v_target, **self._critic_fit_params)

        self._update_policy(obs, act, adv, old_log_p)

        # Print fit information
        self._log_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i[:, self._policy_state_mask], act_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy_t(obs_i[:, self._policy_state_mask])
                loss.backward()
                self._optimizer.step()

    def _log_info(self, dataset, x, v_target, old_pol_dist):
        if self._sw:
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            v_pred = torch.tensor(self._V(x), dtype=torch.float)
            v_err = F.mse_loss(v_pred, torch_v_targets)

            logging_ent = self.policy.entropy(x[:, self._policy_state_mask])
            new_pol_dist = self.policy.distribution(x[:, self._policy_state_mask])
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(
                new_pol_dist, old_pol_dist))
            avg_rwd = np.mean(compute_J(dataset))
            L = int(np.round(np.mean(compute_episodes_length(dataset))))

            self._sw.add_scalar('Training Details/Training Reward', avg_rwd, self._iter)
            self._sw.add_scalar('Training Details/Mean Episode Length', L, self._iter)
            self._sw.add_scalar('Training Details/Value-Function Loss', v_err, self._iter)
            self._sw.add_scalar('Training Details/Entropy', logging_ent, self._iter)
            self._sw.add_scalar('Training Details/KL', logging_kl, self._iter)

    def _post_load(self):
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))
