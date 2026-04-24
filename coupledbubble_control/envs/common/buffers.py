from __future__ import annotations
import torch

from typing import Tuple


class BaseBuffer:
    def __init__(self, shape, dtype, device) -> None:
        # shape := (num_envs, features)
        self.buffer = torch.zeros(shape, dtype=dtype, device=device)

    def update_at(self, indices: torch.Tensor, new_data: torch.Tensor, **kwargs):
        dtype = self.buffer.dtype
        device = self.buffer.device
        new_data = new_data.to(dtype=dtype, device=device)
        self.buffer[indices] = new_data

    def update(self, new_data: torch.Tensor, **kwargs):
        dtype = self.buffer.dtype
        device = self.buffer.device
        new_data = new_data.to(dtype=dtype, device=device)
        self.buffer[:] = new_data

    def get(self):
        return self.buffer  # (num_envs, num_bubble)

class PersistentBuffer(BaseBuffer):
    def __init__(self, shape, dtype, device) -> None:
        """Represents constant variables observed within an episode, such as target
        positions (XT) or equilibrium radii (R0), or cases where historical values
        are not accumulated."""
        super().__init__(shape, dtype, device)

    def get_stacked(self) -> torch.Tensor:
        return self.buffer


class RingBuffer(BaseBuffer):
    def __init__(
            self,
            capacity: int,
            shape: Tuple,
            dtype: torch.dtype,
            device: str    ) -> None:
        
        """
        Stores observed quantities while maintaining a stacked history of past values.
        """
        super().__init__((capacity, *shape), dtype, device)
        self.capacity = capacity
        self.ptr      = 0

    def update(self, new_data: torch.Tensor, *, fill: bool = False):
        dtype = self.buffer.dtype
        device = self.buffer.device
        new_data = new_data.to(dtype=dtype, device=device)
        if fill:
            self.buffer[:, :] = new_data
        else:
            self.buffer[self.ptr] = new_data
            self.ptr = (self.ptr + 1) % self.capacity

    def update_at(self, indices: torch.Tensor, new_data: torch.Tensor, fill: bool = False):
        dtype = self.buffer.dtype
        device = self.buffer.device
        new_data = new_data.to(dtype=dtype, device=device)
        indices = indices.to(dtype=torch.long, device=device)
        if fill:
            # Fill the entire buffer row
            self.buffer[:, indices] = new_data
        else:
            # Update the most recent value
            self.buffer[self.ptr, indices] = new_data
    
    def get(self):
        ptr = (self.ptr - 1) % self.capacity
        return self.buffer[ptr, :]


    def get_stacked(self) -> torch.Tensor:
        # 1. Lekérjük az indexeket (ring buffer logikával)
        idx = torch.arange(self.ptr, self.ptr + self.capacity) % self.capacity
        
        # 2. Visszarendezzük a buffert: 
        # Jelenleg: [capacity, num_envs, num_bubbles]
        # Cél:     [num_envs, num_bubbles, capacity]
        # A permute itt nem másol, csak a stride-okat állítja át:
        stacked = self.buffer[idx].permute(1, 2, 0)
        # Flatten: [num_envs, num_bubbles * capacity]
        return stacked.reshape(stacked.shape[0], -1)
    

class ObservationBufferContainer:
    def __init__(self, 
                 num_envs,
                 observation_space_scheme, 
                 dtype: torch.dtype = torch.float32,
                 device: str = "cuda") -> None:
        self.components     = observation_space_scheme.components
        self.buffers        = {}
        self._build_buffers(num_envs, dtype, device)

    def _build_buffers(self, num_envs, dtype, device):
        for comp in self.components:
            count = len(comp.idx)
            if comp.stack  == 1:
                self.buffers[comp.name] = \
                    PersistentBuffer(
                        shape = (num_envs, count),
                        dtype = dtype,
                        device = device
                )
            else:
                self.buffers[comp.name] = \
                    RingBuffer(
                        capacity = comp.stack,
                        shape = (num_envs, count),
                        dtype = dtype,
                        device= device
                    )

    def update_buffer(self, buffer_name, new_data, *, fill:bool = False):
        self.buffers[buffer_name].update(new_data, fill=fill)

    def update_buffer_at(self, buffer_name, indices, new_data, *, fill: bool = False):
        self.buffers[buffer_name].update_at(indices, new_data, fill=fill)

    def get(self, buffer_name):
        return self.buffers[buffer_name].get()

    def get_stacked(self):
        return torch.cat(
            [buf.get_stacked() for buf in self.buffers.values()],
            dim = 1
        )

