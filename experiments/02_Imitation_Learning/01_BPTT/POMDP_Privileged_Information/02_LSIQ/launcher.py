from itertools import product
from experiment_launcher import Launcher

from experiment_launcher.utils import is_local

if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 10

    launcher = Launcher(exp_name='lsiq_bptt',
                        exp_file='lsiq_experiments',
                        n_seeds=N_SEEDS,
                        n_cores=4,
                        memory_per_core=3000,
                        days=5,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=200,
                          n_steps_per_epoch=5000,
                          n_eval_episodes=10,
                          n_steps_per_fit=1,
                          n_epochs_save=10,
                          logging_iter=10000,
                          gamma=0.99,
                          use_cuda=USE_CUDA,
                          tau=0.005,
                          use_target=True,
                          loss_mode_exp="fix",
                          regularizer_mode="plcy",
                          treat_absorbing_states__=True,
                          learnable_alpha=True)


    alpha_confs = ((1.0, 3e-4),)
    alpha_conf_hiddens = ((0.01, 3e-4),)

    hidden_state_dim_per_layers = [32]

    plcy_loss_modes = ["q_old_policy"]
    lossQ_types = ["sqil_like"]
    truncation_lengths = [5, 32]

    envs = ["HumanoidTorque4Ages.walk.all.perfect", "HumanoidTorque4Ages.run.all.perfect", "Atlas.carry.perfect"]

    for env, hidden_state_dim_per_layer, plcy_loss_mode, alpha_conf,\
        alpha_conf_hidden, lossQ_type, truncation_length in product(envs,
                                                 hidden_state_dim_per_layers,
                                                 plcy_loss_modes,
                                                 alpha_confs,
                                                 alpha_conf_hiddens,
                                                 lossQ_types,
                                                 truncation_lengths):

        init_alpha, lr_alpha = alpha_conf
        init_alpha_hidden, lr_alpha_hidden = alpha_conf_hidden

        # experiment
        launcher.add_experiment(env__=env,
                                plcy_loss_mode__=plcy_loss_mode,
                                hidden_state_dim_per_layer__=hidden_state_dim_per_layer,
                                init_alpha__=init_alpha,
                                lr_alpha__=lr_alpha,
                                init_alpha_hidden__=init_alpha_hidden,
                                lr_alpha_hidden__=lr_alpha_hidden,
                                Q_exp_loss__="MSE",
                                reg_mult__=0.5,
                                lossQ_type__=lossQ_type,
                                truncation_length__=truncation_length,
                                **default_params)

    launcher.run(LOCAL, TEST)
