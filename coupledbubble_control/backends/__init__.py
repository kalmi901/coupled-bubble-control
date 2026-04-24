from .cuda_opts import CUDAOpts
from enum import Enum

class BackendName(str, Enum):
    CUPY        = "cupy"
    NUMBA       = "numba"

class KernelVariant(str, Enum):
    WARP       = "warp"
    SHARED     = "shared"




