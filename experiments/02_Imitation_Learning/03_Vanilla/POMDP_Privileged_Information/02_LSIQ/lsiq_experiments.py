import os
from time import perf_counter, time
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from tqdm import trange

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger

from loco_mujoco import LocoEnv

from s2pg.algorithms import LSIQ
from s2pg.networks.vanilla_networks import SACActorNetwork, SACCriticNetwork, FullyConnectedNetwork
from s2pg.utils.training import do_evaluation, do_evaluation_deterministic, BestAgentSaver


from experiment_launcher import run_experiment


def experiment(env: str = "HumanoidTorque",
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 100,
               logging_iter: int = 100,
               use_cuda: bool = False,
               lr_critic: float = 3e-4,
               lr_actor: float = 3e-5,
               results_dir: str = "./logs",
               plcy_loss_mode: str = "value",
               regularizer_mode: str = "exp_and_plcy",
               reg_mult: float = 0.5,
               Q_exp_loss: str = "MSE",
               n_fits: int = 1,
               loss_mode_exp: str = "fix",
               log_std_min: float = -5.0,
               log_std_max: float = 2.0,
               learnable_alpha: bool = False,
               use_target: bool = True,
               init_alpha: float = 0.001,
               lr_alpha: float = 0.003,
               tau: float = 0.005,
               delay_Q: int = 1,
               lossQ_type: str = "sqil_like",
               treat_absorbing_states: bool = True,
               gamma: float = 0.99,
               seed: int = 0,):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)

    results_dir = os.path.join(results_dir, str(seed))

    # MDP
    mdp = LocoEnv.make(env)
    expert_data = mdp.create_dataset()
    policy_state_mask = mdp.get_mask(("env_type")) if "Humanoid" in env else mdp.get_mask(("weight"))

    # logging stuff
    tb_writer = SummaryWriter(log_dir=results_dir)
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    # Settings
    initial_replay_size = 25000
    max_replay_size = 1000000
    batch_size = 256  # the real batch size is double the size as an expert batch is going to be added
    warmup_transitions = 26000

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0

    target_entropy = None

    # Approximator
    actor_output_shape = (2 * mdp.info.action_space.shape[0],)
    actor_params = dict(network=SACActorNetwork,
                        input_shape=(np.sum(policy_state_mask),),
                        output_shape=actor_output_shape,
                        n_features=256,
                        use_cuda=use_cuda)

    # setup critic
    input_shape_critic = (mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size,
                         input_shape=input_shape_critic,
                         n_features=[256, 256],  # only used when custom_fcn is chosen as critic_architecture
                         activations=["relu", "relu", "identity"],
                         squeeze_out=False,
                         # only used when custom_fcn is chosen as critic_architecture
                         output_shape=(1,),
                         use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor, 'weight_decay': weight_decay_actor}}

    Q_max = 1 / reg_mult * (1 / (1 - gamma))
    Q_min = -1 / reg_mult * (1 / (1 - gamma))

    # create IQfO agent
    agent = LSIQ(mdp_info=mdp.info, batch_size=batch_size, initial_replay_size=initial_replay_size,
                 max_replay_size=max_replay_size, sw=tb_writer, use_target=use_target, Q_max=Q_max, Q_min=Q_min,
                 warmup_transitions=warmup_transitions, tau=tau, lr_alpha=lr_alpha,
                 actor_params=actor_params, actor_optimizer=actor_optimizer, critic_params=critic_params, delay_Q=delay_Q,
                 policy_state_mask=policy_state_mask,
                 target_entropy=target_entropy, critic_fit_params=None, plcy_loss_mode=plcy_loss_mode,
                 regularizer_mode=regularizer_mode, learnable_alpha=learnable_alpha, init_alpha=init_alpha,
                 demonstrations=expert_data, Q_exp_loss=Q_exp_loss,
                 loss_mode_exp=loss_mode_exp, reg_mult=reg_mult, log_std_min=log_std_min, log_std_max=log_std_max,
                 n_fits=n_fits, logging_iter=logging_iter,  lossQ_type=lossQ_type,
                 treat_absorbing_states=treat_absorbing_states, policy_type_params=dict(state_mask=policy_state_mask))

    core = Core(agent, mdp)

    # do initial evaluation
    J, R, L, dataset = do_evaluation(mdp, core, n_eval_episodes, tb_writer, 0, return_dataset=True)
    logger_stoch.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time())

    J, R, L, dataset = do_evaluation_deterministic(mdp, core, n_eval_episodes, tb_writer, 0, return_dataset=True)
    logger_deter.log_numpy(Epoch=0, R_mean=R, J_mean=J, L=L, Time=time())

    # iqfo train loop
    for i in trange(1, n_epochs + 1, 1, leave=False):
        with catchtime() as t:
            # training
            core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, render=False)

            # do the evaluation
            J, R, L, dataset = do_evaluation(mdp, core, n_eval_episodes, tb_writer, i, return_dataset=True)
            logger_stoch.log_numpy(Epoch=i, R_mean=R, J_mean=J, L=L, Time=time())

            J, R, L, dataset = do_evaluation_deterministic(mdp, core, n_eval_episodes, tb_writer, i,
                                                           return_dataset=True)
            logger_deter.log_numpy(Epoch=i, R_mean=R, J_mean=J, L=L, Time=time())

            # save agent if needed
            agent_saver.save(core.agent, J)

    agent_saver.save_curr_best_agent()
    print("Finished.")


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
