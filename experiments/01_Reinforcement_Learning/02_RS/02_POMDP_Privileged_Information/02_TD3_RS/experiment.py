import os
import time
import numpy as np
import torch
from experiment_launcher.decorators import single_experiment
from experiment_launcher import run_experiment
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from mushroom_rl.core import Logger, Core

from s2pg.environments import RandomizedMassGym
from s2pg.policy.recurrent_policy import RecurrentPolicyTD3_preprocess
from s2pg.networks.recurrent_networks import TD3ActorRecurrentNetwork
from s2pg.networks.critics_with_hidden import get_hidden_critic
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic, BestAgentSaver
from s2pg.algorithms import TD3_RS

from tqdm import trange


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
        batch_size: int = 100,
        seed: int = 0,  # This argument is mandatory
        results_dir: str = './logs',  # This argument is mandatory
        use_cuda: bool = False,
        std: float = 0.02,
        std_hidden: float = 0.02,
        rnn_type: str ="gru",
        hidden_state_dim_per_layer: int = 16,
        num_hidden_layers: int = 2,
        critic_architecture: str = "v1",
        obs_to_hide: str = "velocities",
        use_prev_action: bool = True,
        new_state_update_tau: float = 0.0,
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
    mdp = RandomizedMassGym(env, horizon=horizon, gamma=gamma, obs_to_hide=())
    if "Humanoid" in env:
        obs_to_hide = (obs_to_hide, "com_inertia", "com_velocity", "actuator_forces")
    env_mask_policy = mdp.get_mask(obs_to_hide)
    policy_state_mask = np.concatenate([env_mask_policy,
                                        np.ones(hidden_state_dim_per_layer*num_hidden_layers, dtype=bool)])

    actor_params = dict(network=TD3ActorRecurrentNetwork,
                        input_shape=(np.sum(env_mask_policy),),
                        output_shape=mdp.info.action_space.shape,
                        dim_env_state=np.sum(env_mask_policy),
                        n_features=128,
                        n_hidden_features=hidden_state_dim_per_layer,
                        num_hidden_layers=num_hidden_layers,
                        action_scaling=mdp.info.action_space.high[0],
                        rnn_type=rnn_type,
                        use_prev_action=use_prev_action)

    policy_params = dict(std_action=std,
                         std_hidden_state=std_hidden,
                         dim_hidden_state=hidden_state_dim_per_layer*num_hidden_layers,
                         low=mdp.info.action_space.low,
                         high=mdp.info.action_space.high,
                         state_mask=policy_state_mask)

    # setup critic
    if use_prev_action:
        input_shape_critic = (mdp.info.observation_space.shape[0] + 2*mdp.info.action_space.shape[0] +
                              2 * hidden_state_dim_per_layer*num_hidden_layers,)
    else:
        input_shape_critic = (mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0] +
                              2 * hidden_state_dim_per_layer*num_hidden_layers,)

    critic_params = dict(network=get_hidden_critic(critic_architecture),
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr':           lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size,
                         input_shape=input_shape_critic,
                         n_features=256 if critic_architecture == "v3_prev_action" else [512, 256, 256], # only used when custom_fcn is chosen as critic_architecture
                         activations=["relu", "relu", "relu", "identity"],  # only used when custom_fcn is chosen as critic_architecture
                         squeeze_out=True,
                         output_shape=(1,),
                         state_dim=mdp.info.observation_space.shape[0],
                         action_dim=mdp.info.action_space.shape[0],
                         hidden_state_dim=hidden_state_dim_per_layer*num_hidden_layers,
                         use_cuda=use_cuda,
                         dim_env_state=mdp.info.observation_space.shape[0],
                         use_prev_action=use_prev_action
                         )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': 0.0}},
                      policy_delay=2,
                      warmup_transitions=int(0),
                      policy_state_mask=policy_state_mask,
                      noise_std=0.2,
                      noise_clip=0.5,
                      noise_hidden_std=std_hidden*0.5,
                      noise_hidden_clip=std_hidden*1.25,
                      batch_size=batch_size,
                      initial_replay_size=10000,
                      max_replay_size=int(1e6),
                      hidden_state_dim=num_hidden_layers*hidden_state_dim_per_layer,
                      tau=0.005,
                      new_state_update_tau=new_state_update_tau,
                      )

    # Create the agent
    agent = TD3_RS(mdp_info=mdp.info, policy_class=RecurrentPolicyTD3_preprocess, policy_params=policy_params,
                   actor_params=actor_params, sw=sw, logging_iter=10000,
                   critic_params=critic_params, **alg_params)

    # Create Core
    core = Core(agent, mdp)

    # do initial evaluation
    J, R, L = do_evaluation(mdp, core, n_epochs_eval, sw, 0)
    logger_stoch.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time.time())

    J, R, L = do_evaluation_deterministic(mdp, core, n_epochs_eval, sw, 0)
    logger_deter.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time.time())

    for i in trange(1, n_epochs+1, 1, leave=False):
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, render=False)

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
