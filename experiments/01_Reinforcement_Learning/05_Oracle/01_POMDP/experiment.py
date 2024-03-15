import os
import numpy as np
import torch
import time
from experiment_launcher.decorators import single_experiment
from experiment_launcher import run_experiment
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from mushroom_rl.core import Logger, Core
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from s2pg.networks.vanilla_networks import PPONetwork
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic, BestAgentSaver
from s2pg.algorithms import PPO
from s2pg.environments import Gym
from s2pg.policy import PPOBasePolicy


from tqdm import trange


@single_experiment
def experiment(
        env: str = 'Swimmer-v3',
        horizon: int = 1000,
        gamma: float = 0.99,
        n_epochs: int = 300,
        n_steps_per_epoch: int = 1000,
        n_steps_per_fit: int = 1000,
        n_epochs_eval: int = 10,
        n_epochs_save: int = 25,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        batch_size_actor: int = 32,
        batch_size_critic: int = 32,
        n_epochs_policy: int = 10,
        clip_eps_ppo: float = 0.2,
        gae_lambda: float = 0.95,
        seed: int = 0,  # This argument is mandatory
        results_dir: str = './logs',  # This argument is mandatory
        use_cuda: bool = False,
        std_0: float = 1.0
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
    mdp = Gym(env, horizon=horizon, gamma=gamma, obs_to_hide=())

    policy = PPOBasePolicy(network=PPONetwork,
                           input_shape=mdp.info.observation_space.shape,
                           output_shape=mdp.info.action_space.shape,
                           std_0=std_0,
                           n_features=128,
                           use_cuda=use_cuda)

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

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': 0.0}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=batch_size_actor,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      sw=sw
                      )

    # Create the agent
    agent = PPO(mdp_info=mdp.info, policy=policy, critic_params=critic_params, **alg_params)

    # Add preprocessor
    standardizer = StandardizationPreprocessor(mdp.info)
    agent.add_preprocessor(standardizer)

    # Plotter
    # plotter = PlotDataset(mdp.info, obs_normalized=False)

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
