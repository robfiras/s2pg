from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='td3_rs_pI',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=4,
                    memory_per_core=2500,
                    days=2,
                    hours=0,
                    minutes=0,
                    seconds=0,
                    use_timestamp=True
                    )

defaults = dict(n_epochs=200,
                n_steps_per_epoch=10000,
                n_steps_per_fit=1,
                n_epochs_save=25,
                gamma=0.99,
                critic_architecture__="v3_prev_action",
                batch_size=100,
                obs_to_hide="velocities")


# --- Ant ---
launcher.add_experiment(env__="AntPOMDP-v3",
                        std_hidden__=0.1,
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        use_prev_action__=True,
                        new_state_update_tau__=1.0,
                        lr_critic__=1e-3,
                        lr_actor__=1e-3,
                        std__=0.1,
                        **defaults)

# --- HalfCheetah ---
launcher.add_experiment(env__="HalfCheetahPOMDP-v3",
                        std_hidden__=0.01,
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        use_prev_action__=True,
                        new_state_update_tau__=1.0,
                        lr_critic__=1e-3,
                        lr_actor__=1e-3,
                        std__=0.4,
                        **defaults)

# --- Hopper ---
launcher.add_experiment(env__="HopperPOMDP-v3",
                        std_hidden__=0.01,
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        use_prev_action__=True,
                        new_state_update_tau__=1.0,
                        lr_critic__=1e-3,
                        lr_actor__=1e-3,
                        std__=0.1,
                        **defaults)

# --- Humanoid ---
launcher.add_experiment(env__="HumanoidPOMDP-v3",
                        std_hidden__=0.01,
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        use_prev_action__=True,
                        new_state_update_tau__=1.0,
                        lr_critic__=1e-3,
                        lr_actor__=1e-4,
                        std__=0.1,
                        **defaults)

# --- Walker2d ---
launcher.add_experiment(env__="Walker2dPOMDP-v3",
                        std_hidden__=0.1,
                        rnn_type__="gru",
                        hidden_state_dim_per_layer__=32,
                        num_hidden_layers__=1,
                        use_prev_action__=True,
                        new_state_update_tau__=1.0,
                        lr_critic__=1e-3,
                        lr_actor__=1e-3,
                        std__=0.1,
                        **defaults)

launcher.run(LOCAL, TEST)
