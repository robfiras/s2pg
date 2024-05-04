from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='ppo_bptt_pomdp',
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

defaults = dict(n_epochs=100,
                n_steps_per_epoch=50000,
                n_steps_per_fit=2000,
                n_epochs_save=25,
                gamma=0.99,
                n_epochs_eval=10,
                std_0__=0.5,
                obs_to_hide__="velocities")


n_hidden_featuress = [128]
truncation_lengths = [5, 32]

for truncation_length, n_hidden_features in product(truncation_lengths, n_hidden_featuress):

    launcher.add_experiment(env__="HumanoidPOMDP-v3", lr_actor__=1e-4, lr_critic__=1e-4, clip_eps_ppo__=0.2, truncation_length__=truncation_length, n_hidden_features__=n_hidden_features, **defaults)

    launcher.add_experiment(env__="HalfCheetahPOMDP-v3", lr_actor__=5e-5, lr_critic__=5e-5, clip_eps_ppo__=0.2, truncation_length__=truncation_length, n_hidden_features__=n_hidden_features, **defaults)

    launcher.add_experiment(env__="Walker2dPOMDP-v3", lr_actor__=1e-4, lr_critic__=1e-4, clip_eps_ppo__=0.2, truncation_length__=truncation_length, n_hidden_features__=n_hidden_features, **defaults)

    launcher.add_experiment(env__="HopperPOMDP-v3", lr_actor__=5e-5, lr_critic__=5e-5, clip_eps_ppo__=0.1, truncation_length__=truncation_length, n_hidden_features__=n_hidden_features, **defaults)

    launcher.add_experiment(env__="AntPOMDP-v3", lr_actor__=1e-4, lr_critic__=1e-4, clip_eps_ppo__=0.2, truncation_length__=truncation_length, n_hidden_features__=n_hidden_features, **defaults)

launcher.run(LOCAL, TEST)
