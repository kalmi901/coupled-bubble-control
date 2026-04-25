from __future__ import annotations
import torch
from enum import Enum
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class SpaceData:
    shape: Tuple[int, ...]
    dtype: Optional[torch.dtype] = None
    # Optional, specific field
    n: Optional[int] = None
    low: Optional[torch.Tensor] = None
    high: Optional[torch.Tensor] = None


class BaseVectorSpace(ABC):
    def __init__(self, device, num_envs: int = 1, seed: Optional[int] = None):
        self.num_envs = num_envs
        self._seed     = seed
        self._device   = device
        self._generator = torch.Generator(device=self._device)
        if self._seed is not None:
            self._generator.manual_seed(self._seed)
    
    @property
    @abstractmethod
    def space_data(self) -> SpaceData:
        """ Returns the space data describing the Space"""
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """ Generate a random sample"""
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.num_envs, ) + self.space_data.shape
    

class Discrete(BaseVectorSpace):
    def __init__(self, device, n: int, num_envs: int = 1, seed:Optional[int] = None, start: int = 0):
        super().__init__(device, num_envs, seed)
        self._n = n
        self._start = start
        self._data = SpaceData(shape=(1,), dtype=torch.long, n=n)

    def sample(self, device: Optional[str] = None) -> torch.Tensor:
        target_device = device if device is not None else self._device
        return torch.randint(low=self._start,
                            high=self._n,
                            size=self.shape,
                            generator=self._generator,
                            device=target_device)
    
    @property
    def space_data(self) -> SpaceData:
        return self._data

class Box(BaseVectorSpace):
    def __init__(self, device, 
                 low: Union[float, torch.Tensor],
                 high: Union[float, torch.Tensor],
                 n: int = 1,
                 dtype: Optional[torch.dtype] = None,
                 num_envs: int = 1, seed: int | None = None):
        super().__init__(device, num_envs, seed)
        
        dtype = torch.float32 if dtype is None else dtype
        
        def to_tensor(val, n, dtype):
            if isinstance(val, torch.Tensor):
                val = val.to(dtype = dtype)
                n = len(val)
            elif isinstance(val, (int, float)):
                    val = torch.full((n, ), val, dtype=dtype)
            else:
                raise TypeError(f"Unsupported type for bound: {type(val)}")
            return val, n

        low, n_low = to_tensor(low, n, dtype)
        high, n_high = to_tensor(high, n, dtype)

        if n_high != n_low:
            raise RuntimeError(
                f"Bound size mismatch: 'low' has size {n_low}, but 'high' has size {n_high}. "
                "Both bounds must have the same number of dimensions."
            )
        
        self._data = SpaceData(shape=(n_high, ), dtype=dtype, n=n,
                               low = low, high = high)
        
        self._scale  = (high - low) * 0.5
        self._rscale = 1.0 / self._scale
        self._bias   = (high + low) * 0.5

    def sample(self, device: Optional[str] = None) -> torch.Tensor:
        target_device = device if device is not None else self._device
        low, high = self._data.low.to(target_device), self._data.high.to(target_device)  # type: ignore
        return torch.rand(size=self.shape,
                          dtype=self._data.dtype,
                          generator=self._generator,
                          device=target_device) * (high - low) + low
    

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._bias.to(x.device)) * self._rscale.to(x.device)
    
    @property
    def space_data(self) -> SpaceData:
        return self._data
    


    

        
if __name__ == "__main__":
    b = Box(device="cuda",
            low=0,
            high=5,
            n=2,
            dtype=torch.float32,
            num_envs=8,
            seed = 12
            )
    
    print(b.sample())
    print(b.space_data)