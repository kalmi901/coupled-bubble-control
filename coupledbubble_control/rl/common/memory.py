from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, Optional

class RolloutBuffer:
    def __init__(
            self,
            num_envs: int,
            rollout_steps: int,
            single_observation_space_shape: Tuple,
            single_action_space_shape: Tuple,
            device: str,
            dtype: torch.dtype = torch.float64
    ) -> None:
        
        self.num_envs = num_envs
        self.buffer_size = rollout_steps
        self.device = device
        self.dtype = dtype

        # Data Containers
        self.observations = torch.zeros((rollout_steps, num_envs) + single_observation_space_shape, dtype=self.dtype, device=self.device)
        self.actions      = torch.zeros((rollout_steps, num_envs) + single_action_space_shape, dtype=self.dtype, device=self.device)
        self.logprobs     = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=device)
        self.rewards      = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)
        self.values       = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)
        self.dones        = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)

        self.advantages   = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)
        self.returns      = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)

    def store_transtitions(
            self,
            step: int,
            observations: torch.Tensor,
            actions: torch.Tensor,
            logprobs: torch.Tensor,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor
    ) -> None:
        
        if step >= self.buffer_size:
            raise IndexError(
                f"Invalid step value: expected step =< buffer_size "
                f"(received step={step}, buffer_size={self.buffer_size})."
            )
        
        self.observations[step] = observations.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.actions[step]      = actions.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.logprobs[step]     = logprobs.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.rewards[step]      = rewards.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.values[step]       = values.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.dones[step]        = dones.to(dtype=self.dtype, device=self.device, non_blocking=True)

    def compute_gae_estimate_(
            self,
            last_values: torch.Tensor,
            last_dones: torch.Tensor,
            gamma: float,
            gae_lambda: float) -> None:
        
        last_values = last_values.to(self.device, dtype=self.dtype)
        last_dones  = last_dones.to(self.device, dtype=self.dtype)

        lastgaelam = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size -1:
                nextnonterminal = 1.0 - last_dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            self.advantages[t] = lastgaelam

        self.returns = self.advantages + self.values

    def reset_(self) -> None:
        self.observations.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.dones.zero_()

        self.advantages.zero_()
        self.returns.zero_()

    def sample(self,
               target_device: Optional[str]= None):
        
        device = target_device if target_device is not None else self.device
        T, N = self.buffer_size, self.num_envs

        # Flatten observations and actions safely
        b_obs        = self.observations.reshape(T * N, -1)
        b_actions    = self.actions.reshape(T * N, -1)

        # Scalars
        b_logprobs   = self.logprobs.reshape(T * N)
        b_advantages = self.advantages.reshape(T * N)
        b_returns    = self.returns.reshape(T * N)
        b_values     = self.values.reshape(T * N)
        b_dones      = self.dones.reshape(T * N)

        # Move to target device only once per tensor
        if device != self.device:
            b_obs        = b_obs.to(device)
            b_actions    = b_actions.to(device)
            b_logprobs   = b_logprobs.to(device)
            b_advantages = b_advantages.to(device)
            b_returns    = b_returns.to(device)
            b_values     = b_values.to(device)
            b_dones      = b_dones.to(device)

        return (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            b_dones,
        )
