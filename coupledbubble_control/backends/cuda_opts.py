from dataclasses import dataclass, field
from typing import Optional, Literal, List


@dataclass
class CUDAOpts:
    # High-level switch
    mode: Literal["debug", "profile", "release"] = "release"
    max_registerts: Optional[int] = None      # 64, 128, 192 etc
    fastmath: bool = False

    # Common
    lineinfo: bool = False
    
    # Numba-specific:
    opt: bool = True
    debug: bool = False
    cache: bool = False
    
    # CUPY-specific:
    xptxas_verbose: bool = False
    warn_spills: bool = False
    compiler: Literal["nvrtc", "nvcc"] = "nvrtc"

    def __post_init__(self):
        """ Apply high-level profile swithc """
        if self.mode == "debug":
            self.debug = True
            self.lineinfo = True
            self.xptxas_verbose = True
            self.warn_spills = True
            self.fastmath = False

        elif self.mode == "profile":
            self.debug = False
            self.lineinfo = True
            self.xptxas_verbose = True
            self.warn_spills = True
            self.fastmath = False

        elif self.mode == "release":
            self.debug = False
            self.lineinfo = False
            self.xptxas_verbose = False
            self.warn_spills = False
            self.fastmath = False
            self.cache = True

    def to_numba_kwargs(self):
        return {
            "fastmath"  : self.fastmath,
            "opt"       : self.opt,
            "debug"     : self.debug,
            "cache"     : self.cache,
            "max_registers" : self.max_registerts
        }
    
    def to_cupy_options(self):
        opts = []

        if self.fastmath:
            opts.append("--use_fast_math")

        if self.lineinfo:
            opts.append("-lineinfo")

        if self.max_registerts:
            opts.append(f"-maxrregcount={self.max_registerts}")
        
        if self.xptxas_verbose:
            opts.append("-Xptxas=-v")

        if self.warn_spills:
            opts.append("-Xptxas=-warn-spills")

        return opts
