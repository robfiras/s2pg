
![S2PG](img/logo.png)

This is the official code base of the paper [Time-Efficient Reinforcement Learning with Stochastic Stateful Policies](https://arxiv.org/abs/2311.04082), which was presented at the eleventh International Conference on Learning Representations (ICLR 2024) in Vienna Austria.

![humanoid](https://github.com/robfiras/s2pg/assets/69359729/fec5ddd2-e50c-4b3b-af83-d27ad7e31f96)

---

## Method

Stateful policies play an important role in reinforcement learning, such as handling
partially observable environments, enhancing robustness, or imposing an induc-
tive bias directly into the policy structure. The conventional method for training
stateful policies is Backpropagation Through Time (BPTT), which comes with
significant drawbacks, such as slow training due to sequential gradient propagation and the occurrence of vanishing or exploding gradients. The gradient is often
truncated to address these issues, resulting in a biased policy update. We present
a novel approach for training stateful policies by decomposing the latter into a
stochastic internal state kernel and a stateless policy, jointly optimized by following the **Stochastic Stateful Policy Gradient (S2PG)**. We introduce different versions of the stateful
policy gradient theorem, enabling us to easily instantiate stateful variants of popular reinforcement learning and imitation learning algorithms.
We evaluate our approach on complex continuous control tasks, e.g. humanoid locomotion, and demonstrate that our gradient estimator scales effectively
with task complexity while offering a faster and simpler alternative to BPTT.

### Key Advantages
✅ Recurrent version of popular RL algorithms such as PPO, SAC, and TD3.\
✅ Also, implements recurrent versions of imitation learning algorithms such as GAIL and LSIQ.\
✅ Simple to implement with just a few lines of code on-top of non-recurrent algorithms.\
✅ Suiteable for training ODEs within policies.\
✅ Unbiased policy gradient estimator in comparison to window and bptt methods.\
✅ Achieves very robust results across humanoid embodiments.

---

## Installation

You can install this repo by cloning and then

```shell
cd s2pg
pip install -e .
```

The RL experiments are based on OpenAI-Gym v3, which still uses `mujoco_py`. Hence, Mujoco has to be installed manually
to run the RL experiments (see [here](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)). 

To use the imitation learning environments, you need to download the datasets for [LocoMuJoCo](https://github.com/robfiras/loco-mujoco).
While LocoMuJoCo will be installed for you during the installation above, you still need to download the datasets.
To do so, just run this pre-installed command:

```shell
loco-mujoco-download
```
---

## Experiments
All experiment files can be found here to reprocude the results in the paper. Each experiment
has a launcher file, which can be used to run the experiment on a local machine or on a SLURM cluster.

Here is an example on how to run the experiment on a local machine:

```shell
cd experiments/01_Reinforcement_Learning/02_RS/01_POMDP/
python launcher.py
```

Each experiment produces logging files in a directory named `logs`. The latter includes tensorboard logging files.
To visualize the results, you can run tensorboard in the `logs` directory:

```shell
tensorboard --logdir /path/to/logs
```


---

## Citation
```
@inproceedings{alhafez2024,
title={Time-Efficient Reinforcement Learning with Stochastic Stateful Policies},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={Twelfth International Conference on Learning Representations (ICLR)},
year={2024},
url={https://arxiv.org/abs/2311.04082#}}
```
