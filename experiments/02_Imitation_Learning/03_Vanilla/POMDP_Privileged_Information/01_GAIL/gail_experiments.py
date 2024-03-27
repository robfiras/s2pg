import os
from copy import deepcopy
from time import perf_counter
from contextlib import contextmanager
from tqdm import trange
from tempfile import NamedTemporaryFile
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loco_mujoco import LocoEnv

from mushroom_rl.core import Core, Logger

from s2pg.networks.vanilla_networks import PPONetwork
from s2pg.networks.discriminator import DiscriminatorNetwork
from s2pg.policy import PPOBasePolicy
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic, BestAgentSaver
from s2pg.algorithms import GAIL

from experiment_launcher import run_experiment


def experiment(env: str = "HumanoidTorque",
               horizon: int = 1000,
               gamma: float = 0.99,
               n_epochs: int = 300,
               n_steps_per_epoch: int = 1000,
               n_steps_per_fit: int = 5,
               n_epochs_eval: int = 10,
               n_epochs_save: int = 25,
               lr_actor: float = 0.001,
               lr_critic: float = 0.001,
               lrD: float = 3e-4,
               batch_size_actor: int = 32,
               batch_size_critic: int = 32,
               n_epochs_policy: int = 10,
               clip_eps_ppo: float = 0.2,
               gae_lambda: float = 0.95,
               seed: int = 0,  # This argument is mandatory
               results_dir: str = './logs',  # This argument is mandatory
               use_cuda: bool = False,
               std_0: float = 0.02,
               discr_only_state: bool = True,
               use_next_states: bool = True,
               discrim_ent_coef: float = 1e-3,
               discrim_mode: str = "fully_observable",
               train_D_n_th_epoch: int = 3,
               disc_only_state: bool = False,
               env_reward_frac: float = 0.0,
               use_noisy_targets: bool = False):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))
    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)
    sw = SummaryWriter(log_dir=results_dir)

    # MDP
    mdp = LocoEnv.make(env)
    expert_data = mdp.create_dataset()
    policy_state_mask = mdp.get_mask(("env_type")) if "Humanoid" in env else mdp.get_mask(("weight"))

    # logging
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    mdp_info = deepcopy(mdp.info)

    # create the policy
    policy = PPOBasePolicy(network=PPONetwork,
                           input_shape=(np.sum(policy_state_mask),),
                           output_shape=mdp.info.action_space.shape,
                           std_0=std_0,
                           n_features=128,
                           use_cuda=use_cuda,
                           state_mask=policy_state_mask)

    # setup critic
    critic_params = dict(network=PPONetwork,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr':           lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size_critic,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,),
                         n_features=128,
                         use_cuda=use_cuda,
                         )

    discrim_obs_dim = mdp_info.observation_space.shape[0] if discrim_mode == "fully_observable" \
        else np.sum(policy_state_mask)
    discrim_act_mask = [] if disc_only_state else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (discrim_obs_dim + len(discrim_act_mask),)
    discriminator_params = dict(optimizer={'class':  optim.Adam,
                                           'params': {'lr':           lrD,
                                                      'weight_decay': 0.0}},
                                batch_size=2000,
                                network=DiscriminatorNetwork,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                squeeze_out=False,
                                n_features=[512, 256],
                                initializers=None,
                                activations=['tanh', 'tanh', 'identity'],
                                use_actions=False if disc_only_state else True,
                                use_cuda=use_cuda)

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': 0.0}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=batch_size_actor,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      train_D_n_th_epoch=train_D_n_th_epoch,
                      loss_type="logistic_regression",
                      loss_params=dict(entcoeff=discrim_ent_coef),
                      use_noisy_targets=use_noisy_targets,
                      use_next_states=use_next_states,
                      policy_state_mask=policy_state_mask,
                      env_reward_frac=env_reward_frac,
                      discrim_mode=discrim_mode,
                      act_mask=discrim_act_mask)

    agent = GAIL(mdp_info=mdp_info, policy=policy, sw=sw,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=expert_data, **alg_params)

    core = Core(agent, mdp)

    # do initial evaluation
    J, R, L, dataset = do_evaluation(mdp, core, n_epochs_eval, sw, 0, return_dataset=True)
    logger_stoch.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time.time())

    J, R, L, dataset = do_evaluation_deterministic(mdp, core, n_epochs_eval, sw, 0, return_dataset=True)
    logger_deter.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time.time())

    for i in trange(1, n_epochs+1, 1, leave=False):
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, render=False)

        # do the evaluation
        J, R, L, dataset = do_evaluation(mdp, core, n_epochs_eval, sw, i, return_dataset=True)
        logger_stoch.log_numpy(Epoch=i, R_mean=R, J_mean=J, L=L, Time=time.time())

        J, R, L, dataset = do_evaluation_deterministic(mdp, core, n_epochs_eval, sw, i, return_dataset=True)
        logger_deter.log_numpy(Epoch=i, R_mean=R, J_mean=J, L=L, Time=time.time())

        # save agent if needed
        agent_saver.save(core.agent, R)

    agent_saver.save_curr_best_agent()


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    run_experiment(experiment)
