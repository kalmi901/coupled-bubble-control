from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, TypeVar
import torch

from .spaces import Box

@dataclass
class SpaceComponent:
    name: str
    idx: List[int]
    min: List[float]
    max: List[float]

    def get_bounds(self):
        low = self.min
        high = self.max
        return torch.as_tensor(low), torch.as_tensor(high)

@dataclass
class ObservationComponent(SpaceComponent):
    stack: int = 1      # Defualt 1: no value stacking (1 scalar is stored)

    def get_bounds(self):
        low = self.min * self.stack
        high = self.max * self.stack
        return torch.as_tensor(low), torch.as_tensor(high)

@dataclass
class ActionComponent(SpaceComponent):
    scale: float = 1.0

class SpaceSchema:
    components: List[SpaceComponent]

    def build_space(self, num_envs, device, **kwargs) -> Box:
        lows, highs = zip(*[c.get_bounds() for c in self.components])
        return Box(
            device = device, 
            low=torch.cat(lows), 
            high=torch.cat(highs), 
            num_envs=num_envs, **kwargs)
    

class ActionMapingManager:
    def __init__(self, action_space_schema, control_layout) -> None:
        self.components = action_space_schema.components
        self.control_layout = control_layout
        self._feat_col  = {}
        self._act_cols  = []
        self._dev_cols  = []
        self._scales    = []
        self._build_maps()

    def _build_maps(self):
        offset = 0
        for comp in self.components:
            entry = self.control_layout.get(comp.name)
            if entry is None:
                raise KeyError(f"Missing control layout entry for action component '{comp.name}'")
            count = len(comp.idx)

            self._feat_col[comp.name] = range(offset, offset + count)
            self._act_cols.extend(range(offset, offset + count))
            self._dev_cols.extend([entry.base_offset + i for i in comp.idx])
            self._scales.extend([comp.scale] * count)
            offset += count






    




        