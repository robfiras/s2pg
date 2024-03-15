import os
import time
import numpy as np
from experiment_launcher.decorators import single_experiment
from experiment_launcher import run_experiment
from tqdm import trange
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Logger, Core

from s2pg.algorithms import SAC_RS
from s2pg.environments import RandomizedMassGym
from s2pg.policy.recurrent_policy import SACRecurrentPolicy
from s2pg.networks.critics_with_hidden import get_hidden_critic
from s2pg.networks.recurrent_networks import SACRecurrentActorNetwork
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic
from s2pg.utils import BestAgentSaver


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
        rnn_type: str ="gru",
        hidden_state_dim_per_layer: int = 16,
        num_hidden_layers: int = 2,
        critic_architecture: str = "v1",
        obs_to_hide: str = "velocities",
        init_alpha: float = 1.0,
        lr_alpha: float = 3e-4,
        init_alpha_hidden: float = 1.0,
        lr_alpha_hidden: float = 3e-4,
        rate_change_hidden_state: float = 1.0,
        new_state_update_tau: float = 0.0,
        use_prev_action: bool = True,
        action_use_next_hidden: bool = False,
        Q_fit: int = 1,
        add_hidden_entropy: bool = True
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

    actor_output_shape = (2*mdp.info.action_space.shape[0] + num_hidden_layers*hidden_state_dim_per_layer,)
    actor_params = dict(network=SACRecurrentActorNetwork,
                        input_shape=mdp.info.observation_space.shape,
                        output_shape=actor_output_shape,
                        dim_action=mdp.info.action_space.shape[0],
                        dim_env_state=np.sum(env_mask_policy),
                        n_features=256,
                        n_hidden_features=hidden_state_dim_per_layer,
                        num_hidden_layers=num_hidden_layers,
                        rnn_type=rnn_type,
                        rate_change_hidden_state=rate_change_hidden_state,
                        use_prev_action=use_prev_action,
                        action_use_next_hidden=action_use_next_hidden)

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
                      warmup_transitions=10000,
                      hidden_state_dim=hidden_state_dim_per_layer*num_hidden_layers,
                      action_dim=mdp.info.action_space.shape[0],
                      batch_size=batch_size,
                      initial_replay_size=5000,
                      max_replay_size=500000,
                      tau=0.001,
                      lr_alpha=lr_alpha,
                      lr_alpha_hidden=lr_alpha_hidden,
                      init_alpha=init_alpha,
                      init_alpha_hidden=init_alpha_hidden,
                      sw=sw,
                      logging_iter=10000,
                      policy_state_mask=policy_state_mask,
                      policy_type=SACRecurrentPolicy,
                      policy_type_params=dict(state_mask=policy_state_mask),
                      new_state_update_tau=new_state_update_tau,
                      Q_fits=Q_fit,
                      add_hidden_entropy=add_hidden_entropy
                      )

    # Create the agent
    agent = SAC_RS(mdp_info=mdp.info, actor_params=actor_params, critic_params=critic_params, **alg_params)

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
