from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='bptt_sac_pI',
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
                lr_critic=3e-4,
                lr_actor=1e-4,
                gamma=0.99,
                batch_size=256,
                obs_to_hide="velocities")

rnn_types = ["gru"]
hidden_state_dim_per_layers = [32]
num_hidden_layerss = [1]
envs = ["HalfCheetahPOMDP-v3", "AntPOMDP-v3", "HopperPOMDP-v3", "Walker2dPOMDP-v3"]

alpha_confs = ((1.0, 3e-4),)

rate_change_hidden_states = [1.0]
truncation_lengths = [5, 32]

for env, rnn_type, hidden_state_dim_per_layer, num_hidden_layers, alpha_conf,\
    rate_change_hidden_state, truncation_length in product(envs, rnn_types, hidden_state_dim_per_layers,
                                                           num_hidden_layerss, alpha_confs, rate_change_hidden_states,
                                                           truncation_lengths):
    init_alpha, lr_alpha = alpha_conf
    launcher.add_experiment(env__=env,
                            rnn_type__=rnn_type,
                            hidden_state_dim_per_layer__=hidden_state_dim_per_layer,
                            num_hidden_layers__=num_hidden_layers,
                            init_alpha__=init_alpha,
                            lr_alpha__=lr_alpha,
                            rate_change_hidden_state__=rate_change_hidden_state,
                            truncation_length__=truncation_length,
                            **defaults)


envs = ["HumanoidPOMDP-v3"]
alpha_confs = ((1.0, 5e-6),)
for env, rnn_type, hidden_state_dim_per_layer, num_hidden_layers, alpha_conf,\
    rate_change_hidden_state, truncation_length in product(envs, rnn_types, hidden_state_dim_per_layers,
                                                           num_hidden_layerss, alpha_confs, rate_change_hidden_states,
                                                           truncation_lengths):
    init_alpha, lr_alpha = alpha_conf
    launcher.add_experiment(env__=env,
                            rnn_type__=rnn_type,
                            hidden_state_dim_per_layer__=hidden_state_dim_per_layer,
                            num_hidden_layers__=num_hidden_layers,
                            init_alpha__=init_alpha,
                            lr_alpha__=lr_alpha,
                            rate_change_hidden_state__=rate_change_hidden_state,
                            truncation_length__=truncation_length,
                            **defaults)

launcher.run(LOCAL, TEST)
