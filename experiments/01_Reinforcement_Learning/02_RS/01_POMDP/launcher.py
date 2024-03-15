from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25
JOBLIB_PARALLEL_JOBS = 1


launcher = Launcher(exp_name='ppo_rs_pomdp',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=4,
                    memory_per_core=2500,
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
                n_epochs_eval=10,
                obs_to_hide="velocities",
                std_0__=0.5,
                gamma=0.99,
                lr_actor__=5e-5,
                lr_critic__=5e-5,
                )


n_hidden_featuress = [128]
n_critic_featuress = [128]

for n_hidden_features, n_critic_features in product(n_hidden_featuress, n_critic_featuress):

    launcher.add_experiment(env__="HumanoidPOMDP-v3", clip_eps_ppo__=0.1, n_hidden_features__=n_hidden_features,
                            n_critic_features__=n_critic_features, **defaults)

    launcher.add_experiment(env__="HalfCheetahPOMDP-v3", clip_eps_ppo__=0.2, n_hidden_features__=n_hidden_features,
                            n_critic_features__=n_critic_features, **defaults)

    launcher.add_experiment(env__="Walker2dPOMDP-v3", clip_eps_ppo__=0.05, n_hidden_features__=n_hidden_features,
                            n_critic_features__=n_critic_features, **defaults)

    launcher.add_experiment(env__="HopperPOMDP-v3", clip_eps_ppo__=0.05, n_hidden_features__=n_hidden_features,
                            n_critic_features__=n_critic_features, **defaults)

    launcher.add_experiment(env__="AntPOMDP-v3", clip_eps_ppo__=0.2, n_hidden_features__=n_hidden_features,
                            n_critic_features__=n_critic_features, **defaults)

launcher.run(LOCAL, TEST)



