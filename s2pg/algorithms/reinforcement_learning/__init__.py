# vanilla
from .deep_actor_critic import PPO, DDPG, TD3, SAC

# bptt_pI
from .deep_actor_critic import PPO_BPTT, DDPG_BPTT, TD3_BPTT, SAC_BPTT

# bptt privileged information
from .deep_actor_critic import PPO_BPTT_pI, DDPG_BPTT_pI, TD3_BPTT_pI, SAC_BPTT_pI

# rs
from .deep_actor_critic import PPO_RS, DDPG_RS, TD3_RS, SAC_RS

# all policy gradient methods
from .policy_gradient import REINFORCE_RS, BatchedREINFORCE_RS, BatchedREINFORCE_BPTT,\
    StatefulPolicyGradient, BatchedStatefulPolicyGradient
