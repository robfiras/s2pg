from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='ppo_window_pomdp',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=1,
                    memory_per_core=2000,
                    days=1,
                    hours=0,
                    minutes=0,
                    seconds=0,
                    use_timestamp=True
                    )

defaults = dict(n_epochs=100,
                n_steps_per_epoch=50000,
                n_steps_per_fit=2000,
                n_epochs_save=25,
                gamma=0.99,
                num_hidden_layers=1,
                n_epochs_eval=10,
                obs_to_hide="velocities")

truncation_lengths = [5, 32]
n_featuress = [128]

for truncation_length, n_features in product(truncation_lengths, n_featuress):

    launcher.add_experiment(env__="HalfCheetahPOMDP-v3", lr_actor__=5e-5, truncation_length__=truncation_length,
                            lr_critic__=5e-5, std_0__=0.5, clip_eps_ppo__=0.2, n_features__=n_features,
                            **defaults)

    launcher.add_experiment(env__="Walker2dPOMDP-v3", lr_actor__=1e-4, truncation_length__=truncation_length,
                            lr_critic__=1e-4, std_0__=0.5, clip_eps_ppo__=0.2, n_features__=n_features,
                            **defaults)

    launcher.add_experiment(env__="HopperPOMDP-v3", lr_actor__=5e-5, truncation_length__=truncation_length,
                            lr_critic__=5e-5, std_0__=0.5, clip_eps_ppo__=0.1, n_features__=n_features,
                            **defaults)

    launcher.add_experiment(env__="AntPOMDP-v3", lr_actor__=1e-4, truncation_length__=truncation_length,
                            lr_critic__=1e-4, std_0__=0.5, clip_eps_ppo__=0.2, n_features__=n_features,
                            **defaults)

    launcher.add_experiment(env__="HumanoidPOMDP-v3", lr_actor__=5e-5, truncation_length__=truncation_length,
                            lr_critic__=5e-5, std_0__=0.5, clip_eps_ppo__=0.1, n_features__=n_features,
                            **defaults)

launcher.run(LOCAL, TEST)
