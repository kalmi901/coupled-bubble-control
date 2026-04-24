from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple

@dataclass(frozen=True)
class ModelSpec:
    name: str           # Model Name
    sd: int             # (Unit) System Dimension
    nup: int            # Number of Unit specific parameters
    nsp: int            # Number of Shared parameters
    ncm: int            # Number Coupling Matrixes
    nk: int             # Number of Harmonic Components
    ne: int             # Number of Events
    ncf: int            # Number of Coupling Factors
    nct: int            # Number of Coupling Terms
    ac: Literal["CONST", "SW_N", "SW_A"]    # Harmonic forcing type
    ncp: int = field(init=False)            # 4 x NK (PA, FREQ, PS, WAVENUM)

    def __post_init__(self):
        # Numbef of control parameters
        #ndp = (4 if self.AC in ("SW_N", "SW_A") else 3) * self.NK
        ncp = 4 * self.nk
        object.__setattr__(self, "ncp", ncp)


@dataclass(frozen=True)
class ExecutionSpec:
    ups: int                    # Unit per system
    ns: int = 1                 # Number of systems (min 1)
    spb: int = 1                # System per block  (min 1)
    ndo: int = 0                # Number of dense output stored (0 -> no dense output is stored)

    tile: int = field(init=False)     # Unit per system padded to closest power of 2

    total_threads: int = field(init=False)      # Number of threads
    active_threads: int = field(init=False)     # Number of active threads
    block_size: int = field(init=False)         # Threads per block
    grid_size: int = field(init=False)          # Number of block

    def __post_init__(self) -> None:
        if self.ups < 1:
            raise ValueError("ups must be > 1")
        if self.ns < 1:
            raise ValueError("ns must be >= 1")
        if self.spb < 1:
            raise ValueError("spb must be >= 1")
        object.__setattr__(self, "tile", 1 << (self.ups - 1).bit_length())

        object.__setattr__(self, "total_threads", self.ns * self.tile)
        object.__setattr__(self, "active_threads", self.ns * self.ups)
        object.__setattr__(self, "block_size", self.spb * self.tile)
        object.__setattr__(self, "grid_size", (self.ns + self.spb - 1) // self.spb )


@dataclass(frozen=True)
class SolverSpec:
    # -- ODE SOLVER PARAMETERS --
    atol: float = 1e-9
    rtol: float = 1e-9
    min_step: float = 1e-16
    max_step: float = 1e-1
    growth_limit: float = 2.0
    shrink_limit: float = 0.5
    #max_steps: int = 1000_0000

    # -- LINSOLVE PARAMETERS --
    lin_atol: float = 1e-9
    lin_rtol: float = 1e-9
    lin_max_iter: int = 100
    lin_relaxation: float = 1.0


# Helpers to map layour entries
@dataclass(frozen=True)
class LayoutEntry:
    group: str
    base_offset : int

    








