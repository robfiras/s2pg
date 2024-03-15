from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='sac_rs_pI',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=4,
                    memory_per_core=1500,
                    days=4,
                    hours=0,
                    minutes=0,
                    seconds=0,
                    use_timestamp=True
                    )

defaults = dict(n_epochs=200,
                n_steps_per_epoch=10000,
                n_steps_per_fit=1,
                n_epochs_save=25,
                lr_actor=1e-4,
                gamma=0.99,
                action_use_next_hidden=True,
                obs_to_hide="velocities",
                critic_architecture__="v3_prev_action",
                add_hidden_entropy__=False,
                batch_size=256)

# --- Ant ---
launcher.add_experiment(env__="AntPOMDP-v3",
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        init_alpha__=1.0,
                        lr_alpha__=0.0003,
                        init_alpha_hidden__=0.01,
                        lr_alpha_hidden__=5e-6,
                        new_state_update_tau__=1.0,
                        use_prev_action__=True,
                        lr_critic__=0.0003,
                        **defaults)


# --- HalfCheetah ---
launcher.add_experiment(env__="HalfCheetahPOMDP-v3",
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        init_alpha__=1.0,
                        lr_alpha__=0.0003,
                        init_alpha_hidden__=0.01,
                        lr_alpha_hidden__=0.0003,
                        new_state_update_tau__=1.0,
                        use_prev_action__=True,
                        lr_critic__=0.001,
                        **defaults)


# --- Hopper ---
launcher.add_experiment(env__="HopperPOMDP-v3",
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        init_alpha__=1.0,
                        lr_alpha__=0.0003,
                        init_alpha_hidden__=0.01,
                        lr_alpha_hidden__=0.0003,
                        new_state_update_tau__=1.0,
                        use_prev_action__=True,
                        lr_critic__=0.001,
                        **defaults)

# --- Humanoid ---
launcher.add_experiment(env__="HumanoidPOMDP-v3",
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        init_alpha__=1.0,
                        lr_alpha__=5e-06,
                        init_alpha_hidden__=0.01,
                        lr_alpha_hidden__=5e-06,
                        new_state_update_tau__=1.0,
                        use_prev_action__=True,
                        lr_critic__=0.0003,
                        **defaults)

# --- Walker2d ---
launcher.add_experiment(env__="Walker2dPOMDP-v3",
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        init_alpha__=1.0,
                        lr_alpha__=0.0003,
                        init_alpha_hidden__=0.01,
                        lr_alpha_hidden__=0.0003,
                        new_state_update_tau__=1.0,
                        use_prev_action__=True,
                        lr_critic__=0.0003,
                        **defaults)

launcher.run(LOCAL, TEST)
