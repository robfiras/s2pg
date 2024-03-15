import os
import time 
import numpy as np
import torch
from experiment_launcher.decorators import single_experiment
from experiment_launcher import run_experiment
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from mushroom_rl.core import Logger, Core

from s2pg.networks.recurrent_networks import PPOActorBPTTNetwork
from s2pg.networks.vanilla_networks import PPONetwork_prev_a
from s2pg.environments import Gym
from s2pg.policy import StatefulPolicyPPO_joint_prev_a
from s2pg.algorithms import PPO_RS
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic, BestAgentSaver

from tqdm import trange


def get_POMDP_params(pomdp_type):
    if pomdp_type == "no_velocities":
        return dict(obs_to_hide=("velocities",), random_force_com=False)
    elif pomdp_type == "no_positions":
        return dict(obs_to_hide=("positions",), random_force_com=False)
    elif pomdp_type == "windy":
        return dict(obs_to_hide=tuple(), random_force_com=True)


@single_experiment
def experiment(
        env: str = 'Swimmer-v3',
        horizon: int = 1000,
        gamma: float = 0.99,
        n_epochs: int = 300,
        n_steps_per_epoch: int = 1000,
        n_steps_per_fit: int = 5,
        n_epochs_eval: int = 10,
        n_epochs_save: int = 25,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001,
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
        n_hidden_features: int = 128,
        num_hidden_layers: int = 1,
        n_critic_features: int = 128,
        obs_to_hide: str = "no_velocities"
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # prepare logging
    results_dir = os.path.join(results_dir, str(seed))
    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)
    sw = SummaryWriter(log_dir=results_dir)
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    # MDP
    if "Humanoid" in env:
        obs_to_hide = (obs_to_hide, "com_inertia", "com_velocity")
    mdp = Gym(env, horizon=horizon, gamma=gamma, obs_to_hide=obs_to_hide)
    
    # create the cpg policy
    dim_env_state = mdp.info.observation_space.shape[0]
    dim_action = mdp.info.action_space.shape[0]

    policy = StatefulPolicyPPO_joint_prev_a(network=PPOActorBPTTNetwork,
                                            input_shape=(dim_env_state, ),
                                            output_shape=(dim_action+n_hidden_features*num_hidden_layers,),
                                            n_features=128,
                                            rnn_type=rnn_type,
                                            n_hidden_features=n_hidden_features,
                                            num_hidden_layers=num_hidden_layers,
                                            dim_env_state=mdp.info.observation_space.shape[0],
                                            hidden_state_dim=n_hidden_features,
                                            dim_action=dim_action,
                                            action_dim=dim_action,
                                            std_0=std_0,
                                            mdp_info=mdp.info,
                                            )

    # setup critic
    input_shape_critic = (mdp.info.observation_space.shape[0]+n_hidden_features+mdp.info.action_space.shape[0],)
    critic_params = dict(network=PPONetwork_prev_a,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr':           lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size_critic,
                         input_shape=input_shape_critic,
                         output_shape=(1,),
                         n_features=n_critic_features,
                         use_cuda=use_cuda,
                         )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': 0.0}},
                      n_epochs_policy=n_epochs_policy,
                      hidden_state_dim=n_hidden_features*num_hidden_layers,
                      batch_size=batch_size_actor,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      sw=sw
                      )

    # Create the agent
    agent = PPO_RS(mdp_info=mdp.info, policy=policy, critic_params=critic_params,
                   include_prev_a_in_V=True, **alg_params)

    # Create Core
    core = Core(agent, mdp)

    # do initial evaluation
    J, R, L = do_evaluation(mdp, core, n_epochs_eval, sw, 0)
    logger_stoch.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time.time())

    J, R, L = do_evaluation_deterministic(mdp, core, n_epochs_eval, sw, 0)
    logger_deter.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time.time())

    for i in trange(1, n_epochs+1, 1, leave=False):
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit)

        # do the evaluation
        J, R, L = do_evaluation(mdp, core, n_epochs_eval, sw, i)
        logger_stoch.log_numpy(Epoch=i, R_mean=R, J_mean=J, L=L, Time=time.time())

        J, R, L = do_evaluation_deterministic(mdp, core, n_epochs_eval, sw, i)
        logger_deter.log_numpy(Epoch=i, R_mean=R, J_mean=J, L=L, Time=time.time())

        # save agent if needed
        agent_saver.save(core.agent, J)

    agent_saver.save_curr_best_agent()


if __name__ == '__main__':
    run_experiment(experiment)
