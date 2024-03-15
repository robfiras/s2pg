from itertools import product
from experiment_launcher import Launcher

from experiment_launcher.utils import is_local

if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 10

    launcher = Launcher(exp_name='lsiq_rs',
                        exp_file='lsiq_experiments',
                        n_seeds=N_SEEDS,
                        n_cores=4,
                        memory_per_core=3000,
                        days=3,
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
                          learnable_alpha=True)

    log_std = [(-5, 2)]

    hidden_state_dim_per_layers = [32]
    use_prev_actions = [False]
    adapt_tau_on_experts = [False]
    alpha_confs = ((1.0, 3e-4), (1.0, 3e-5))
    alpha_conf_hiddens = ((0.01, 3e-4))
    lrs = ((3e-4, 1e-4))
    gradient_pen_coeffs = [0.0]
    plcy_loss_modes = ["q_old_policy"]
    lossQ_types = ["sqil_like"]
    treat_absorbing_statess = [True]
    Q_exp_losses = ["MSE"]
    reg_mults = [0.5]

    envs = ["HumanoidTorque4Ages.walk.all.perfect", "HumanoidTorque4Ages.run.all.perfect", "Atlas.carry.perfect"]

    for env, hidden_state_dim_per_layer,\
        use_prev_action, plcy_loss_mode, adapt_tau_on_expert,\
        alpha_conf, alpha_conf_hidden, lr, lossQ_type, treat_absorbing_states, gradient_pen_coeff, Q_exp_loss, reg_mult in product(envs,
                                                 hidden_state_dim_per_layers,
                                                 use_prev_actions,
                                                 plcy_loss_modes,
                                                 adapt_tau_on_experts,
                                                 alpha_confs, alpha_conf_hiddens, lrs, lossQ_types, treat_absorbing_statess,
                                                 gradient_pen_coeffs, Q_exp_losses, reg_mults):
        init_alpha, lr_alpha = alpha_conf
        init_alpha_hidden, lr_alpha_hidden = alpha_conf_hidden
        lr_critic, lr_actor = lr

        # experiment
        launcher.add_experiment(env__=env,
                                plcy_loss_mode__=plcy_loss_mode,
                                adapt_tau_on_expert__=adapt_tau_on_expert,
                                hidden_state_dim_per_layer__=hidden_state_dim_per_layer,
                                use_prev_action__=use_prev_action,
                                Q_exp_loss__=Q_exp_loss,
                                reg_mult__=reg_mult,
                                init_alpha__=init_alpha,
                                init_alpha_hidden__=init_alpha_hidden,
                                lr_alpha__=lr_alpha,
                                lr_alpha_hidden__=lr_alpha_hidden,
                                lr_critic__=lr_critic,
                                lr_actor__=lr_actor,
                                lossQ_type__=lossQ_type,
                                treat_absorbing_states__=treat_absorbing_states,
                                gradient_pen_coeff__=gradient_pen_coeff,
                                **default_params)


    launcher.run(LOCAL, TEST)
