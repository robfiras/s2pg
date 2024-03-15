from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25

launcher = Launcher(exp_name='ppo_vanilla_pomdp',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=1,
                    memory_per_core=2000,
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
                n_epochs_eval=10,
                lr_critic=3e-4,
                lr_actor=3e-4,
                gamma=0.99,
                std_0=1.0)

envs = ["HalfCheetahPOMDP-v3", "Walker2dPOMDP-v3", "HopperPOMDP-v3", "AntPOMDP-v3"]


for env in envs:
    launcher.add_experiment(env__=env, lr_critic__=1e-4, lr_actor__=1e-4, clip_eps_ppo__=0.01, **defaults)


launcher.add_experiment(env__="HumanoidPOMDP-v3", lr_critic__=5e-5, lr_actor__=5e-5, clip_eps_ppo__=0.005, **defaults)

launcher.run(LOCAL, TEST)

