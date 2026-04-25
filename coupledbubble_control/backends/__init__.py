from .cuda_opts import CUDAOpts
from enum import Enum

class BackendName(str, Enum):
    CUPY        = "cupy"
    NUMBA       = "numba"

class KernelVariant(str, Enum):
    WARP       = "warp"
    SHARED     = "shared"


def get_current_device_name() -> str:
    import re

    def sanitize(name: str) -> str:
        # csak betűk + számok, szóközök kuka
        name = re.sub(r"[^a-zA-Z0-9]", "", name)
        # opcionális rövidítés
        name = name.replace("NVIDIA", "")
        name = name.replace("GeForce", "")
        return name

    # CuPy backend
    try:
        import cupy as cp
        name = cp.cuda.Device().name
        return sanitize(name)
    except Exception:
        pass

    # Numba backend
    try:
        from numba import cuda
        name = cuda.get_current_device().name.decode()
        return sanitize(name)
    except Exception:
        pass

    return "unknownGPU"