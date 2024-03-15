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
from s2pg.networks.recurrent_networks import PPOActorBPTTNetwork
from s2pg.policy.recurrent_policy import RecurrentPolicyPPOBPTT_preprocess
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic, BestAgentSaver
from s2pg.algorithms import GAIL_BPTT_pI
from s2pg.utils.math import GailDiscriminatorLoss

from experiment_launcher import run_experiment


def experiment(env: str = "HumanoidTorque",
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
               rnn_type: str ="gru",
               n_hidden_features: int = 32,
               num_hidden_layers: int = 1,
               use_next_states: bool = True,
               discrim_ent_coef: float = 1e-3,
               discrim_mode: str = "fully_observable",
               truncation_length: int = 5, 
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
    env_mask_policy = mdp.get_mask(("env_type")) if "Humanoid" in env else mdp.get_mask(("weight"))
    policy_state_mask = np.concatenate([env_mask_policy,
                                        np.ones(n_hidden_features * num_hidden_layers, dtype=bool)])

    # logging
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    mdp_info = deepcopy(mdp.info)

    # create the policy
    dim_env_state = mdp.info.observation_space.shape[0]
    dim_action = mdp.info.action_space.shape[0]
    hidden_state_dim = n_hidden_features * num_hidden_layers

    policy = RecurrentPolicyPPOBPTT_preprocess(network=PPOActorBPTTNetwork,
                                               input_shape=(dim_env_state,),
                                               output_shape=(dim_action,),
                                               n_features=128,
                                               rnn_type=rnn_type,
                                               n_hidden_features=n_hidden_features,
                                               num_hidden_layers=num_hidden_layers,
                                               dim_env_state=np.sum(env_mask_policy),
                                               dim_hidden_state=n_hidden_features,
                                               dim_action=dim_action,
                                               std_0=std_0,
                                               mdp_info=mdp.info,
                                               state_mask=policy_state_mask
                                               )

    # setup critic
    input_shape_critic = (mdp.info.observation_space.shape[0],)
    critic_params = dict(network=PPONetwork,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr':           lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size_critic,
                         input_shape=input_shape_critic,
                         output_shape=(1,),
                         n_features=128,
                         use_cuda=use_cuda,
                         )

    discrim_obs_dim = mdp_info.observation_space.shape[0] if discrim_mode == "fully_observable" \
        else np.sum(policy_state_mask) - hidden_state_dim
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
                      hidden_state_dim=hidden_state_dim,
                      n_epochs_policy=n_epochs_policy,
                      batch_size=batch_size_actor,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      train_D_n_th_epoch=train_D_n_th_epoch,
                      loss=GailDiscriminatorLoss(entcoeff=discrim_ent_coef),
                      use_noisy_targets=use_noisy_targets,
                      use_next_states=use_next_states,
                      policy_state_mask=policy_state_mask,
                      env_reward_frac=env_reward_frac,
                      act_mask=discrim_act_mask,
                      discrim_mode=discrim_mode,
                      dim_env_state=dim_env_state,
                      truncation_length=truncation_length)

    agent = GAIL_BPTT_pI(mdp_info=mdp_info, policy=policy, sw=sw,
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
        agent_saver.save(core.agent, J)

    agent_saver.save_curr_best_agent()


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    run_experiment(experiment)
