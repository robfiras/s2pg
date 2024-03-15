from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 25


launcher = Launcher(exp_name='sac_vanilla_pI',
                    exp_file='experiment',
                    n_seeds=N_SEEDS,
                    n_cores=4,
                    memory_per_core=1500,
                    days=2,
                    hours=0,
                    minutes=0,
                    seconds=0,
                    use_timestamp=True
                    )

defaults = dict(n_epochs=200,
                n_steps_per_epoch=10000,
                n_steps_per_fit=1,
                n_epochs_save=100,
                lr_actor=1e-4,
                lr_critic=3e-4,
                init_alpha=1.0,
                lr_alpha=5e-6,
                gamma=0.99,
                batch_size=256)


envs = ["HalfCheetahPOMDP-v3", "HopperPOMDP-v3", "Walker2dPOMDP-v3", "HumanoidPOMDP-v3", "AntPOMDP-v3"]

for env in envs:
    alpha_conf = (1.0, 5e-6) if "HumanoidPOMDP" in env else (1.0, 3e-4)
    init_alpha, lr_alpha = alpha_conf
    launcher.add_experiment(env__=env,
                            init_alpha__=init_alpha,
                            lr_alpha__=lr_alpha,
                            **defaults)

launcher.run(LOCAL, TEST)
