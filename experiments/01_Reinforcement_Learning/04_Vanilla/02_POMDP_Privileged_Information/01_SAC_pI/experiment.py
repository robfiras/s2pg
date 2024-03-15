import os
import time
import numpy as np
import torch
from experiment_launcher.decorators import single_experiment
from experiment_launcher import run_experiment
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from mushroom_rl.core import Logger, Core

from s2pg.utils import BestAgentSaver
from s2pg.networks.vanilla_networks import SACActorNetwork, SACCriticNetwork
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic
from s2pg.environments import RandomizedMassGym
from s2pg.algorithms import SAC

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
        obs_to_hide: str = "velocities",
        init_alpha: float = 1.0,
        lr_alpha: float = 3e-4,
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
    policy_state_mask = mdp.get_mask(obs_to_hide)

    actor_params = dict(network=SACActorNetwork,
                        n_features=256,
                        input_shape=(np.sum(policy_state_mask),),
                        output_shape=(mdp.info.action_space.shape[0]*2,),
                        use_cuda=use_cuda)

    critic_input_shape = (mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=SACCriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic, 'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         n_features=256,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': 0.0}},
                      warmup_transitions=10000,
                      batch_size=batch_size,
                      initial_replay_size=5000,
                      max_replay_size=500000,
                      tau=0.001,
                      lr_alpha=lr_alpha,
                      init_alpha=init_alpha,
                      sw=sw,
                      logging_iter=1000,
                      policy_state_mask=policy_state_mask,
                      )

    # Create the agent
    agent = SAC(mdp_info=mdp.info, actor_params=actor_params, critic_params=critic_params, **alg_params)

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
