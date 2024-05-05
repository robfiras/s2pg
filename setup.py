from setuptools import setup, find_packages

requires_list = ["torchdiffeq", "mushroom_rl>=1.10.0", "tensorboard", "experiment_launcher", "loco-mujoco",
                 "gym==0.24.1"]

setup(name='s2pg',
      version='0.1',
      description='Implementation of stochastic stateful policies (e.g., RNN\'s or ODE\'s) for reinforcement and'
                  'imitation learning based on Mushroom-rl.',
      license='MIT',
      author="Firas Al-Hafez",
      author_mail="fi.alhafez@gmail.com",
      packages=[package for package in find_packages()
                if package.startswith('s2pg')],
      install_requires=requires_list,
      zip_safe=False,
      )
