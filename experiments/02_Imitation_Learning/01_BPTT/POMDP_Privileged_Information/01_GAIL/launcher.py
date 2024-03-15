import os
from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 10

    launcher = Launcher(exp_name='gail_bptt',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=4,
                        memory_per_core=3000,
                        days=5,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True)

    default_params = dict(n_epochs=200,
                          n_steps_per_epoch=100000,
                          n_epochs_eval=10,
                          n_epochs_save=50,
                          n_steps_per_fit=1000,
                          n_hidden_features=32,
                          num_hidden_layers=1,
                          rnn_type="gru",
                          use_cuda=USE_CUDA)

    discrim_ent_coefs = [1e-3]
    train_D_n_th_epoch = [1]
    use_next_states = [False]
    stds = [0.5]
    learning_rates = [1e-5]
    clip_eps_ppos = [0.005]
    discrim_modes = ["fully_observable"]
    truncation_lengths = [5, 32]

    envs = ["HumanoidTorque4Ages.walk.all.perfect", "HumanoidTorque4Ages.run.all.perfect", "Atlas.carry.perfect"]

    for env, d_ent_coef, train_Dth, usn, std, lr,\
        clip_eps_ppo, discrim_mode, truncation_length in product(envs, discrim_ent_coefs, train_D_n_th_epoch, use_next_states,
                                stds, learning_rates, clip_eps_ppos, discrim_modes, truncation_lengths):

        use_noisy_targets = True if d_ent_coef == 0.0 else False
        launcher.add_experiment(**default_params,
                                env__=env,
                                use_noisy_targets__=use_noisy_targets,
                                discrim_ent_coef__=d_ent_coef,
                                train_D_n_th_epoch=train_Dth,
                                std_0__=std,
                                lr_actor__=lr,
                                lr_critic__=lr,
                                discrim_mode__=discrim_mode,
                                clip_eps_ppo__=clip_eps_ppo,
                                use_next_states__=usn,
                                truncation_length__=truncation_length)
    launcher.run(LOCAL, TEST)
