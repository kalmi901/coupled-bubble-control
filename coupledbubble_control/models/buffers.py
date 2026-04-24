from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Annotated, TypeAlias, Union, Tuple, Optional
import cupy as cp
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from ..backends import BackendName

# ---- Generic array aliases ----------
HostArray: TypeAlias = np.ndarray
CuArray: TypeAlias = cp.ndarray
CudaArray: TypeAlias = DeviceNDArray
BufferData: TypeAlias = Union[HostArray, CuArray, CudaArray]

TimeArray: TypeAlias = Annotated[BufferData, "(NS,)"]
ScalarFieldArray: TypeAlias = Annotated[BufferData, "(NS, TILE)"]
StateArray: TypeAlias = Annotated[BufferData, "(SD, NS, TILE)"]
UnitParamArray: TypeAlias = Annotated[BufferData, "(NUP, NS, TILE)"]
SystemParamArray: TypeAlias = Annotated[BufferData, "(NSP,NS)"]
ControlParamArray: TypeAlias = Annotated[BufferData, "(NCP, NS)"]
CouplingMatrixArray: TypeAlias = Annotated[BufferData, ("NCM, UPS, NS, TILE")]
DenseTimeArray: TypeAlias = Annotated[BufferData, "(NDO, NS)"]
DenseStateArray: TypeAlias = Annotated[BufferData, "(NDO, SD, NS, TILE)"]
EventFuncArray: TypeAlias = Annotated[BufferData, "(NS,)"]
StatusFlagArray: TypeAlias = Annotated[BufferData, "(NS,)"]
CounterAarry: TypeAlias = Annotated[BufferData, "(NS,)"]

@dataclass(slots=True)
class Buffer:
    pass

@dataclass(slots=True)
class ActualStateBuffer(Buffer):
    actual_time: TimeArray
    time_end: TimeArray
    time_begin: TimeArray
    time_step: TimeArray
    actual_state: StateArray


@dataclass(slots=True)
class ParameterBuffer(Buffer):
    unit_params: UnitParamArray
    system_params: SystemParamArray
    control_params: ControlParamArray
    coupling_matrices: CouplingMatrixArray


@dataclass(slots=True)
class DenseOutputBuffer(Buffer):
    dense_index: TimeArray
    dense_time: DenseTimeArray
    dense_state: DenseStateArray

@dataclass(slots=True)
class StatusBuffer:
    actual_event: EventFuncArray
    status_flags: StatusFlagArray
    total_steps: CounterAarry
    rejected_steps: CounterAarry
    convergece_success: CounterAarry
    convergence_failures: CounterAarry


@dataclass(slots=True)
class BufferSet:
    state: ActualStateBuffer
    params: ParameterBuffer
    dense: DenseOutputBuffer
    status: StatusBuffer


class BufferOps:
    def __init__(self, backend = None) -> None:
        self.backend = backend  # NUMBA, CUPY, None
        # None --> numpy array

    
    # -- alloc not used --
    def alloc_actual_state(self, model_spec, execution_spec):
        return ActualStateBuffer(
            actual_time = self._alloc((execution_spec.ns, ), np.float64),
            time_end = self._alloc((execution_spec.ns, ), np.float64),
            time_begin = self._alloc((execution_spec.ns, ), np.float64),
            time_step   = self._alloc((execution_spec.ns, ), np.float64),
            actual_state= self._alloc((model_spec.sd, execution_spec.ns, execution_spec.tile)),
        )

    def alloc_params(self, model_spec, execution_spec):
        # TODO: placeholder
        return ParameterBuffer(
            unit_params       = 1,
            system_params     = 1,
            control_params    = 1,
            coupling_matrices = 1
        )
    
    def alloc_dense_output(self, model_spec, execution_spec):
        # TODO: placeholder
        return DenseOutputBuffer(
            dense_index = 1,
            dense_time  = 1,
            dense_state = 1
        )

    def zeros(self, shape: Tuple[int, ...], dtype=np.float64) -> BufferData:
        pass

    
    def sync_to_device(self, src_buf, dst_buf, stream=None):
        """ Host -> Device copy between existing buffers"""
        for field in fields(src_buf):
            src = getattr(src_buf, field.name)
            dst = getattr(dst_buf, field.name)
            self._copy_h2d(src, dst, stream)


    def sync_to_host(self, src_buf, dst_buf, stream=None):
        """ Device -> Host copy between existing buffers """
        for field in fields(src_buf):
            src = getattr(src_buf, field.name)
            dst = getattr(dst_buf, field.name)
            self._copy_d2h(src, dst, stream)


    def create_device_state(self, src_buf, stream=None):
        return ActualStateBuffer(
            actual_time = self.from_host_ary(src_buf.actual_time, stream=stream),
            time_end    = self.from_host_ary(src_buf.time_end, stream=stream),
            time_begin  = self.from_host_ary(src_buf.time_begin, stream=stream),
            time_step   = self.from_host_ary(src_buf.time_step, stream=stream),
            actual_state = self.from_host_ary(src_buf.actual_state, stream=stream)
        )
    
    def create_device_params(self, src_buf, stream=None):
        return ParameterBuffer(
            unit_params     = self.from_host_ary(src_buf.unit_params, stream=stream),
            system_params   = self.from_host_ary(src_buf.system_params, stream=stream),
            control_params  = self.from_host_ary(src_buf.control_params, stream=stream),
            coupling_matrices = self.from_host_ary(src_buf.coupling_matrices, stream=stream),
        )

    def create_device_dense_output(self, src_buf, stream=None):
        return DenseOutputBuffer(
            dense_index = self.from_host_ary(src_buf.dense_index, stream=stream),
            dense_time  = self.from_host_ary(src_buf.dense_time, stream=stream),
            dense_state = self.from_host_ary(src_buf.dense_state, stream=stream),
        )

    def create_device_status(self, src_buf, stream=None):
        return StatusBuffer(
            actual_event = self.from_host_ary(src_buf.actual_event, stream=stream),
            status_flags = self.from_host_ary(src_buf.status_flags, stream=stream),
            # Counters TODO--> these are not used in the solver
            total_steps = self.from_host_ary(src_buf.total_steps, stream=stream),
            rejected_steps = self.from_host_ary(src_buf.rejected_steps, stream=stream),
            convergece_success = self.from_host_ary(src_buf.convergece_success, stream=stream),
            convergence_failures = self.from_host_ary(src_buf.convergence_failures, stream=stream)
        )

    def from_host_ary(self, src, *, stream=None):
        host = np.ascontiguousarray(src, dtype=src.dtype)
        if self.backend == BackendName.NUMBA:
            if stream is None:
                stream = cuda.default_stream()
            return cuda.to_device(host, stream=stream)
        elif self.backend == BackendName.CUPY:
            return cp.asarray(host)
        else:
            return host


    def _copy_h2d(self, src, dst, stream=None):
        if self.backend == BackendName.NUMBA:
            # Numba esetén a cél (dst) egy DeviceNDArray
            dst.copy_to_device(src, stream=stream)
        elif self.backend == BackendName.CUPY:
            # CuPy-nál a .set() a legegyszerűbb host->device-re
            dst.set(src)
        else:
            np.copyto(dst, src)


    def _copy_d2h(self, src, dst, stream=None):
        if self.backend == BackendName.NUMBA:
            # src: DeviceNDarray, dst: numpy array
            src.copy_to_host(dst, stream=stream)
        elif self.backend == BackendName.CUPY:
            src.get(out=dst)
        else:
            np.copyto(dst, src)


    def _alloc(self, shape: Tuple[int, ...], dtype=np.float64) -> BufferData:
        if self.backend == BackendName.NUMBA:
            return cuda.device_array(shape, dtype)
        elif self.backend == BackendName.CUPY:
            return cp.empty(shape, dtype)
    
