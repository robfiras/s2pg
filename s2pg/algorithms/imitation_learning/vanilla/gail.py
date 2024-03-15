from copy import deepcopy

import torch
import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.utils.dataset import parse_dataset

from s2pg.algorithms import PPO
from s2pg.utils.replay_memory import ReplayMemory
from s2pg.utils.math import GailDiscriminatorLoss, LeastSquaresGailDiscriminatorLoss
from s2pg.utils.preprocessors import StandardizationPreprocessor_ext


class GAIL(PPO):
    """
    Generative Adversarial Imitation Learning(GAIL) implementation.

    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, discriminator_params, D_standardizer=None,
                 train_D_n_th_epoch=3, n_epochs_discriminator=1, demonstrations=None, env_reward_frac=0.0,
                 act_mask=None, use_next_states=False, use_noisy_targets=False, discrim_mode="fully_observable",
                 discriminator_fit_params=None, loss_params=None, loss_type="logistic_regression", **kwargs):

        super(GAIL, self).__init__(**kwargs)

        # standardizer
        self._D_standardizer = D_standardizer

        # discriminator params
        self._discriminator_fit_params = (dict() if discriminator_fit_params is None
                                          else discriminator_fit_params)

        if loss_type == "logistic_regression":
            loss_params = dict() if loss_params is None else loss_params
            self._loss = GailDiscriminatorLoss(**loss_params)
        elif loss_type == "least_squares":
            self._loss = LeastSquaresGailDiscriminatorLoss()
        else:
            raise ValueError("Unknown loss type %s" % loss_type)
        self._loss_type = loss_type
        discriminator_params.setdefault("loss", deepcopy(self._loss))
        self._D = Regressor(TorchApproximator, **discriminator_params)
        self._train_D_n_th_epoch = train_D_n_th_epoch
        self._n_epochs_discriminator = n_epochs_discriminator

        self._env_reward_frac = env_reward_frac

        if discrim_mode == "fully_observable":
            self._discrim_state_mask = np.ones(self.mdp_info.observation_space.shape[0], dtype=bool)
        elif discrim_mode == "partially_observable":
            self._discrim_state_mask = deepcopy(self._policy_state_mask)
        else:
            raise ValueError("Unknown discriminator mode %s." % discrim_mode)

        self._replay_memory_expert = ReplayMemory(initial_size=0,
                                                  max_size=len(demonstrations["states"]))

        self._init_expert_replay_memory(demonstrations)

        assert 0.0 <= env_reward_frac <= 1.0, "Environment reward frac must be between [0,1]"
        assert demonstrations is not None or env_reward_frac == 1.0, "No demonstrations have been loaded"

        # select which observations / actions to discriminate
        if not "actions" in demonstrations:
            act_mask = []

        self._act_mask = np.arange(demonstrations["actions"].shape[1]) \
            if act_mask is None else np.array(act_mask, dtype=np.int64)

        self._use_next_state = use_next_states

        self._epoch_counter = 1

        self._use_noisy_targets = use_noisy_targets

        # add the standardization preprocessor
        self._preprocessors.append(StandardizationPreprocessor_ext(self.mdp_info))

        self._add_save_attr(
            discriminator_fit_params='pickle',
            _loss='torch',
            _train_n_th_epoch='pickle',
            _D='mushroom',
            _env_reward_frac='pickle',
            _demonstrations='pickle!',
            _act_mask='pickle',
            _discrim_state_mask='pickle',
            _use_next_state='pickle',
            _use_noisy_targets='pickle',
            _D_standardizer='pickle',
            _train_D_n_th_epoch='pickle',
            _loss_type='primitive'
        )

    def _init_expert_replay_memory(self, demonstrations):

        states = demonstrations["states"]
        actions = demonstrations["actions"]
        rewards = demonstrations["rewards"]
        next_states = demonstrations["next_states"]
        absorbings = demonstrations["absorbing"]
        lasts = demonstrations["last"]

        for s, a, r, ss, ab, last in zip(states, actions, rewards, next_states, absorbings, lasts):

            sample = [s, a, r, ss, ab, last]
            self._replay_memory_expert.add([sample])

    def fit(self, dataset, **info):
        state, action, reward, next_state, absorbing, last = parse_dataset(dataset)
        x = state.astype(np.float32)
        u = action.astype(np.float32)
        r = reward.astype(np.float32)
        xn = next_state.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        # update running mean and std
        if self._standardizer:
            self._standardizer.update_mean_std(obs)

        # create reward
        if self._env_reward_frac < 1.0:

            # create reward from the discriminator(can use fraction of environment reward)
            r_disc = self.make_discrim_reward(x, u, xn)
            r = r * self._env_reward_frac + r_disc * (1 - self._env_reward_frac)

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

        # fit discriminator
        self._fit_discriminator(x, u, xn)

    def _fit_discriminator(self, plcy_obs, plcy_act, plcy_n_obs):
        plcy_obs = plcy_obs[:, self._discrim_state_mask]
        plcy_act = plcy_act[:, self._act_mask]
        plcy_n_obs = plcy_n_obs[:, self._discrim_state_mask]

        if self._iter % self._train_D_n_th_epoch == 0:

            for epoch in range(self._n_epochs_discriminator):

                demo_obs, demo_act, demo_reward, demo_n_obs,\
                    demo_absorbing, demo_last = self._replay_memory_expert.get(plcy_obs.shape[0])

                demo_obs = self.standardize_discrim_states(demo_obs)
                demo_act = demo_act[:, self._act_mask]
                demo_n_obs = self.standardize_discrim_states(demo_n_obs)

                # get batch of data to discriminate
                if self._use_next_state and not self._act_mask.size > 0:
                    input_states = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
                    input_next_states = np.concatenate([plcy_n_obs, demo_n_obs.astype(np.float32)])
                    inputs = (input_states, input_next_states)
                elif self._act_mask.size > 0 and not self._use_next_state:
                    input_states = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
                    input_actions = np.concatenate([plcy_act, demo_act.astype(np.float32)])
                    inputs = (input_states, input_actions)
                elif self._act_mask.size > 0 and self._use_next_state:
                    raise ValueError("Discriminator with states, actions and next states as input currently not supported.")
                else:
                    input_states = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
                    inputs = (input_states,)

                # update running mean if neccessary
                if self._D_standardizer is not None:
                    self._D_standardizer.update_mean_std(np.concatenate([plcy_obs, demo_obs.astype(np.float32)]))

                # create label targets
                if self._use_noisy_targets:
                    demo_target = np.random.uniform(low=0.80, high=0.99, size=(plcy_obs.shape[0], 1)).astype(np.float32)
                    plcy_target = np.random.uniform(low=0.01, high=0.10, size=(plcy_obs.shape[0], 1)).astype(np.float32)
                else:
                    plcy_target = np.zeros(shape=(plcy_obs.shape[0], 1)).astype(np.float32)
                    demo_target = np.ones(shape=(plcy_obs.shape[0], 1)).astype(np.float32)

                targets = np.concatenate([plcy_target, demo_target])

                self._D.fit(*inputs, targets, **self._discriminator_fit_params)

                self._discriminator_logging(inputs, targets)

    def _discriminator_logging(self, inputs, targets):
        if self._sw:
            plcy_inputs, demo_inputs = self.divide_data_to_demo_and_plcy(inputs)
            loss = deepcopy(self._loss)
            loss_eval = loss.forward(to_float_tensor(self._D(*inputs)), torch.tensor(targets))
            self._sw.add_scalar('DiscrimLoss', loss_eval, self._iter // 3)

            # calculate the accuracies
            dout_exp = torch.sigmoid(torch.tensor(self.discrim_output(*demo_inputs, apply_mask=False)))
            dout_plcy = torch.sigmoid(torch.tensor(self.discrim_output(*plcy_inputs, apply_mask=False)))
            accuracy_exp = np.mean((dout_exp > 0.5).numpy())
            accuracy_gen = np.mean((dout_plcy < 0.5).numpy())
            self._sw.add_scalar('D_Generator_Accuracy', accuracy_gen, self._iter // 3)
            self._sw.add_scalar('D_Out_Generator', np.mean(dout_plcy.numpy()), self._iter // 3)
            self._sw.add_scalar('D_Expert_Accuracy', accuracy_exp, self._iter // 3)
            self._sw.add_scalar('D_Out_Expert', np.mean(dout_exp.numpy()), self._iter // 3)

            # calculate individual losses
            plcy_target = targets[0:len(targets)//2]
            demo_target = targets[len(targets)//2:]
            loss_exp = loss.forward(to_float_tensor(self._D(*demo_inputs)), torch.tensor(demo_target)) / 2
            loss_gen = loss.forward(to_float_tensor(self._D(*plcy_inputs)), torch.tensor(plcy_target)) / 2
            self._sw.add_scalar('Generator_loss', loss_gen, self._iter // 3)
            self._sw.add_scalar('Expert_Loss', loss_exp, self._iter // 3)
            if self._loss_type == "logistic_regression":
                bernoulli_ent = torch.mean(loss.logit_bernoulli_entropy(torch.tensor(self.discrim_output(*inputs, apply_mask=False))))
                neg_bernoulli_ent_loss = -loss.entcoeff * bernoulli_ent
                self._sw.add_scalar('Bernoulli Ent.', bernoulli_ent, self._iter // 3)
                self._sw.add_scalar('Neg. Bernoulli Ent. Loss (incl. in DiscrimLoss)', neg_bernoulli_ent_loss, self._iter // 3)

    def divide_data_to_demo_and_plcy(self, inputs):
        if self._act_mask.size > 0:
            input_states, input_actions = inputs
            plcy_obs = input_states[0:len(input_states)//2]
            plcy_act = input_actions[0:len(input_actions)//2]
            plcy_inputs = (plcy_obs, plcy_act)
            demo_obs = input_states[len(input_states)//2:]
            demo_act = input_actions[len(input_actions)//2:]
            demo_inputs = (demo_obs, demo_act)
        elif self._use_next_state:
            input_states, input_next_states = inputs
            plcy_obs = input_states[0:len(input_states)//2]
            plcy_n_obs = input_next_states[0:len(input_next_states)//2]
            plcy_inputs = (plcy_obs, plcy_n_obs)
            demo_obs = input_states[len(input_states)//2:]
            demo_n_obs = input_next_states[len(input_next_states)//2:]
            demo_inputs = (demo_obs, demo_n_obs)
        else:
            input_states = inputs[0]
            plcy_inputs = (input_states[0:len(input_states)//2],)
            demo_inputs = (input_states[len(input_states)//2:],)
        return plcy_inputs, demo_inputs

    def prepare_discrim_inputs(self, inputs, apply_mask=True):
        if self._use_next_state and not self._act_mask.size > 0:
            states, next_states = inputs
            states = states[:, self._discrim_state_mask] if apply_mask else states
            next_states = next_states[:, self._discrim_state_mask] if apply_mask else next_states
            inputs = (states, next_states)
        elif self._act_mask.size > 0 and not self._use_next_state:
            states, actions = inputs
            states = states[:, self._discrim_state_mask] if apply_mask else states
            actions = actions[:, self._act_mask] if apply_mask else actions
            inputs = (states, actions)
        elif self._act_mask.size > 0 and self._use_next_state:
            raise ValueError("Discriminator with states, actions and next states as input currently not supported.")
        else:
            states = inputs[0][:, self._discrim_state_mask] if apply_mask else inputs[0]
            inputs = (states,)
        return inputs

    def standardize_discrim_states(self, states):
        standardizer = self.preprocessors[0]
        states = standardizer.call_without_updating_stats(states)
        return states[:, self._discrim_state_mask]

    def discrim_output(self, *inputs, apply_mask=True):
        inputs = self.prepare_discrim_inputs(inputs, apply_mask=apply_mask)
        d_out = self._D(*inputs)
        return d_out

    @torch.no_grad()
    def make_discrim_reward(self, state, action, next_state, apply_mask=True):
        if self._use_next_state:
            d = self.discrim_output(state, next_state, apply_mask=apply_mask)
        else:
            d = self.discrim_output(state, action, apply_mask=apply_mask)

        if self._loss_type == "logistic_regression":
            plcy_prob = 1/(1 + np.exp(-d))     # sigmoid
            reward = np.squeeze(-np.log(1 - plcy_prob + 1e-8)).astype(np.float32)
        else:
            # least-squares reward
            reward = np.squeeze(np.maximum(0.0, 1.0 - 0.25 * np.square(d - 1)))  # makes reward bounded in [0.0, 1.0]

        return reward
