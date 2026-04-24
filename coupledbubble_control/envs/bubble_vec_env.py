from __future__ import annotations
import torch
from abc import ABC, abstractmethod

from typing import Any, List, Union, Optional

from .common.specs import SpaceSchema


class BubbleVecEnv(ABC):
    model: Any      # The physical model with a method named as `solve()`
    action_space_schema: SpaceSchema
    observaton_space_schema: SpaceSchema
    trajectory_buffer: Optional[Any] = None

    def __init__(self,
                num_envs: int,
                envs_per_block: int,
                action_space_schema: SpaceSchema,
                observation_space_schema: SpaceSchema,
                seed: Optional[int] = None, 
                device: str = "cuda"
                ) -> None:
        super().__init__()

        self.num_envs = num_envs
        self.envs_per_block = min(envs_per_block, num_envs)

        self.action_space = action_space_schema.build_space(num_envs, device, seed=seed)
        self.observation_space = observation_space_schema.build_space(num_envs, device, seed=seed)


    @abstractmethod
    def step(self, action: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, seed: int, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def reset_envs(self, env_ids, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        print("Environment is closed")





