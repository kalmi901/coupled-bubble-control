from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import time
import numpy as np

from typing import Dict, Any, Union, Optional

from .common.memory import RolloutBuffer
from .common.writers import TFWriter
from .common.policies import ActorCriticGaussianPolicy, ActorCriticBetaPolicy


class PPO:
    metadata = {"hyperparameters" : ["pi_learning_rate", "vf_learning_rate", "ft_learning_rate", 
                                     "gamma", "gae_lambda", "mini_batch_size", "clip_coef", "clip_vloss", "ent_coef", "vf_coef", "max_grad_norm",
                                     "target_kl", "norm_adv", "num_envs", "rollout_steps", "num_update_epochs", "gradient_steps",
                                     "seed", "torch_deterministic", "cuda", "buffer_device", 
                                     "policy_type", "hidden_dims", "activations", "shared_dims"]}

    default_net_arch = {
        "hidden_dims": [126, 84],
        "activations": ["ReLU", "ReLU"],
        "shared_dims": 0}
    
    def __init__(
            self,
            venvs: Any,
            learning_rate: float = 2.5e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            mini_batch_size: int = 256,
            clip_coef: float = 0.2,
            clip_vloss: bool = True,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            norm_adv: bool = True,
            rollout_steps: int = 32,
            num_update_epochs: int = 4,
            seed: int = 1,
            torch_deterministic: bool = True,
            cuda: bool = True,
            buffer_device: str = "cuda",
            net_archs: Dict = default_net_arch,
            policy: str = "Beta",
            ) -> None:
        
        # Seeds ------
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        # Attribures ---
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.venvs = venvs
        self.num_envs = venvs.num_envs

        # Hyperparameters ---
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.rollout_steps = rollout_steps
        self.mini_batch_size = mini_batch_size
        self.batch_size = self.num_envs * self.rollout_steps       # = buffer_size
        self.num_update_epochs = num_update_epochs
        self.iterations_per_epoch = self.iterations_per_epoch = self.batch_size // self.mini_batch_size
        self.gradient_steps = self.num_update_epochs * self.iterations_per_epoch
        self.buffer_device = buffer_device

        # Neural Netwokrs ---
        self.policy_type = self.policy_type
        self.hidden_dims = net_archs["hidden_dims"]
        self.activations = net_archs["activations"]
        self.shared_dims = net_archs["shared_dims"]

        action_high = venvs.action_space.space_data.high
        action_low  = venvs.action_space.space_data.low
        action_shape = venvs.action_space.space_data.shape
        observation_shape = venvs.observation_space.space_data.shape

        if self.policy_type == "Gaussian":
            self.policy: nn.Module

        elif self.policy_type == "Beta":
            self.policy: nn.Module

        else:
            raise ValueError(f"Err: policy_type {self.policy_type} is not a valid policy. Please Choose a valid policy `Beta` or `Gaussian`")

        # Optimizer ---
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate)

        print("----Policy----")
        print(self.policy)

        # Rollout buffer ---
        self.memory = RolloutBuffer(
            self.num_envs,
            self.rollout_steps,
            self.venvs.single_observation_space.shape,
            self.venvs.single_action_space.shape,
            self.buffer_device
        )