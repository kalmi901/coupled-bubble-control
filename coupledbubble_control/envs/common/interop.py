from __future__ import annotations
from typing import TypeAlias, Union
import torch
import cupy as cp
from numba.cuda.cudadrv.devicearray import DeviceNDArray


CuArray: TypeAlias = cp.ndarray
CudaArray: TypeAlias = DeviceNDArray
GpuArray: TypeAlias = Union[CuArray, CudaArray]


def create_view(d_ary: GpuArray, name: str, *, hard_fail: bool = True) -> torch.Tensor:
    """
    Create a torch view over a GPU array (CuPy or Numba) with zero-copy expectation.
    """

    if isinstance(d_ary, CuArray):
        out = torch.from_dlpack(d_ary)
        backend = "cupy"
    elif isinstance(d_ary, CudaArray):
        out = torch.as_tensor(d_ary, device="cuda")
        backend = "numba"
    else:
        raise TypeError(
            f"[{name}] Unsupported type: {type(d_ary)}. "
            "Expected cupy.ndarray or numba.cuda.DeviceNDArray."
        )

    # --- sanity checks ---
    _assert_cuda_tensor(out, name)

    assert_zero_copy(d_ary, out, name, backend=backend, hard_fail=hard_fail)

    return out


def assert_zero_copy(
    d_ary: GpuArray,
    torch_tensor: torch.Tensor,
    name: str,
    *,
    backend: str | None = None,
    hard_fail: bool = False,
) -> None:
    """
    Verify that torch tensor shares the same device pointer as backend array.
    """

    ptr_torch = torch_tensor.data_ptr()

    # --- backend pointer detection ---
    if hasattr(d_ary, "device_ctypes_pointer"):  # numba
        ptr_backend = d_ary.device_ctypes_pointer.value
        backend = backend or "numba"
    elif hasattr(d_ary, "data") and hasattr(d_ary.data, "ptr"):  # cupy
        ptr_backend = d_ary.data.ptr
        backend = backend or "cupy"
    else:
        raise TypeError(f"[{name}] Unknown backend type: {type(d_ary)}")

    # --- pointer check ---
    if ptr_backend != ptr_torch:
        msg = (
            f"[Interop WARNING] {name} is NOT zero-copy!\n"
            f"  backend:        {backend}\n"
            f"  backend ptr:    {ptr_backend}\n"
            f"  torch ptr:      {ptr_torch}\n"
            f"  shape:          {tuple(torch_tensor.shape)}\n"
            f"  dtype:          {torch_tensor.dtype}\n"
            f"  device:         {torch_tensor.device}\n"
        )
        if hard_fail:
            raise RuntimeError(msg)
        else:
            print(msg)
    else:
        print(
            f"[Interop OK] {name} zero-copy ({backend}) | "
            f"shape={tuple(torch_tensor.shape)} dtype={torch_tensor.dtype}"
        )


def _assert_cuda_tensor(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise RuntimeError(f"[{name}] Tensor is not on CUDA device")

    if t.device.type != "cuda":
        raise RuntimeError(f"[{name}] Unexpected device: {t.device}")