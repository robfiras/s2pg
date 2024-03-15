from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='td3_bptt_pI',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=4,
                    memory_per_core=2500,
                    days=7,
                    hours=0,
                    minutes=0,
                    seconds=0,
                    use_timestamp=True
                    )

defaults = dict(n_epochs=200,
                n_steps_per_epoch=10000,
                n_steps_per_fit=1,
                n_epochs_save=25,
                n_epochs_eval=10,
                lr_actor__=0.0005,
                gamma=0.99,
                obs_to_hide="velocities")

lrs = [1e-3]
envs = ["HopperPOMDP-v3", "Walker2dPOMDP-v3", "AntPOMDP-v3", "HumanoidPOMDP-v3"]
truncation_lengths = [5, 32]
hidden_state_dim_per_layers = [32]
num_hidden_layerss = [1]
stds = [0.1]

for env, lr, truncation_length, hidden_state_dim_per_layer,\
    num_hidden_layers, std in product(envs, lrs, truncation_lengths,
                                      hidden_state_dim_per_layers, num_hidden_layerss, stds):
    launcher.add_experiment(env__=env, lr_critic__=lr, 
                            hidden_state_dim_per_layer__=hidden_state_dim_per_layer,
                            num_hidden_layers__=num_hidden_layers, std__=std,
                            truncation_length__=truncation_length, **defaults)

envs = ["HalfCheetahPOMDP-v3"]
stds = [0.4]

for env, lr, truncation_length, hidden_state_dim_per_layer,\
    num_hidden_layers, std in product(envs, lrs, truncation_lengths,
                                      hidden_state_dim_per_layers, num_hidden_layerss, stds):
    launcher.add_experiment(env__=env, lr_critic__=lr,
                            hidden_state_dim_per_layer__=hidden_state_dim_per_layer,
                            num_hidden_layers__=num_hidden_layers, std__=std,
                            truncation_length__=truncation_length, **defaults)

launcher.run(LOCAL, TEST)
