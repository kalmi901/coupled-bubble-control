from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, Any, Callable, Dict
from .buffers import (
    BufferSet,
    ActualStateBuffer,
    ParameterBuffer,
    StatusBuffer,
    DenseOutputBuffer,
    BufferOps)
from .specs import ModelSpec, ExecutionSpec, SolverSpec, LayoutEntry
from ..backends import BackendName, KernelVariant, CUDAOpts


@dataclass
class MaterialProperties:
    PV  = 0.0       # Vapour Pressure [Pa]
    RHO = 998.0     # Liquod Density [kg/m**3]
    ST  = 0.0725    # Surface Tension [N/m]
    VIS = 0.001     # Liquid viscosity [Pa s]
    CL  = 1500      # Liqid Sound Speed
    P0  = 1.013*1e5 # Ambient Pressure [Pa]
    PE  = 1.4       # Polytrophic Exponent [-]


class ModelRuntime():
    def __init__(
            self,
            backend: BackendName,
            variant: KernelVariant,
            stream = None) -> None:
        
        self.backend = backend
        self.variant = variant
        self.stream  = stream

        self.host_buffers: BufferSet
        self.device_buffers: Optional[BufferSet] = None
        self.solver: Optional[Callable] = None
        self.cuda_opts: CUDAOpts

class Model:
    _model_spec: ModelSpec
    _execution_spec: ExecutionSpec
    _solver_spec: SolverSpec
    
    _available_backends: Tuple[BackendName, ...]
    _kernel_variants: Tuple[KernelVariant, ...]
    runtime: ModelRuntime
    buffer_ops: BufferOps

    control_layout: Dict[str, LayoutEntry]
    state_layout: Dict[str, LayoutEntry]
    
    @property
    def time_shape(self):
        return (self._execution_spec.ns, )
    
    @property
    def state_shape(self):
        return (self._model_spec.sd,
                self._execution_spec.ns,
                self._execution_spec.tile)
    
    @property
    def unit_param_shape(self):
        return (self._model_spec.nup,
                self._execution_spec.ns,
                self._execution_spec.tile)
    @property
    def system_param_shape(self):
        return (self._model_spec.nsp,
                self._execution_spec.ns)
    
    @property
    def coupling_matrix_shape(self):
        return (self._model_spec.ncm,
                self._execution_spec.ups,
                self._execution_spec.ns,
                self._execution_spec.tile)
    
    @property
    def control_param_shape(self):
        return (self._model_spec.ncp,
                self._execution_spec.ns)
    
    @property
    def dense_state_shape(self):
        return (self._execution_spec.ndo,
                self._model_spec.sd,
                self._execution_spec.ns,
                self._execution_spec.tile)
    
    @property
    def dense_time_shape(self):
        return (self._execution_spec.ndo,
                self._execution_spec.ns)
    
    @property
    def dense_index_shape(self):
        return (self._execution_spec.ns, )
    
    @property
    def status_buffer_shape(self):
        return (self._execution_spec.ns, )
    
    def solve(self, 
              max_steps: int = 100_000,
              kernel_steps: int = 2048,
              *, 
              sync_to_host: bool = False,
              sync_from_host: bool = False,
              stream = None,
              benchmark: bool = False, debug: bool = False):
        self._ensure_buffers()

        if sync_from_host:
            # State sync
            self.buffer_ops.sync_to_device(
                src_buf = self.runtime.host_buffers.state,
                dst_buf = self.runtime.device_buffers.state,    # type: ignore
                stream  = stream 
            )

            self.buffer_ops.sync_to_device(
                src_buf = self.runtime.host_buffers.dense,
                dst_buf = self.runtime.device_buffers.dense,    # type: ignore
                stream  = stream 
            )         

        if self.runtime.solver is None:
            self.runtime.solver = self._make_solver(
                self.runtime.backend, self.runtime.variant)

        out = self.runtime.solver(
            d_buffers = self.runtime.device_buffers,
            model_spec = self._model_spec,
            execution_spec = self._execution_spec,
            solver_spec = self._solver_spec,
            max_steps = max_steps,
            kernel_steps = kernel_steps,
            benchmark = benchmark,
            debug = debug
        )

        if sync_to_host:
            # State sync
            self.buffer_ops.sync_to_host(
                src_buf = self.runtime.device_buffers.state,    # type: ignore
                dst_buf = self.runtime.host_buffers.state, 
                stream  = stream)
            
            self.buffer_ops.sync_to_host(
                src_buf = self.runtime.device_buffers.dense,    # type: ignore
                dst_buf = self.runtime.host_buffers.dense,
                stream  = stream
            )
            # TODO: status sync

        return out


    def build_parameters(self):
        """
        Must return:
            unit_params
            system_params
            coupling_matrices
        """
        raise NotImplementedError
    
    def build_control_parameters(self):
        raise NotImplementedError
    
    def build_default_initial_state(self):
        raise NotImplementedError


    def _initialize_host_buffers(self):
        unit_params, \
        system_params, \
        coupling_matrices = self.build_parameters()
        control_params = self.build_control_parameters()

        params_buffer = ParameterBuffer(
            unit_params = unit_params,
            system_params = system_params,
            control_params = control_params,
            coupling_matrices = coupling_matrices
        )

        x0 = self.build_default_initial_state()

        state_buffer = ActualStateBuffer(
            actual_time = np.zeros(self.time_shape, dtype=np.float64),
            time_end = np.ones(self.time_shape, dtype=np.float64),
            time_begin = np.zeros(self.time_shape, dtype=np.float64),
            time_step   = np.full(self.time_shape, 1e-6, dtype=np.float64),
            actual_state = x0,
        )

        dense_buffer = DenseOutputBuffer(
            dense_index = np.zeros(self.dense_index_shape, dtype=np.int32),
            dense_time = np.zeros(self.dense_time_shape, dtype=np.float64),
            dense_state = np.zeros(self.dense_state_shape, dtype=np.float64),
        )

        status_buffer = StatusBuffer(
            actual_event = np.full(self.status_buffer_shape, -1.0, dtype=np.float64),   # -1 --> not initialized (should not be zero to avoid early terminatiun due to event)
            status_flags = np.zeros(self.status_buffer_shape, dtype=np.int32),
            # counters
            total_steps = np.zeros(self.status_buffer_shape, dtype=np.int32),
            rejected_steps = np.zeros(self.status_buffer_shape, dtype=np.int32),
            convergece_success = np.zeros(self.status_buffer_shape, dtype=np.int32),
            convergence_failures = np.zeros(self.status_buffer_shape, dtype=np.int32)
        )

        self.runtime.host_buffers = BufferSet(
            state = state_buffer,
            params = params_buffer,
            dense = dense_buffer,
            status = status_buffer
        )


    def _ensure_buffers(self, stream = None):
        if self.runtime.device_buffers is not None:
            return

        self.runtime.device_buffers = BufferSet(
            state=self.buffer_ops.create_device_state(self.runtime.host_buffers.state),
            params=self.buffer_ops.create_device_params(self.runtime.host_buffers.params),
            dense=self.buffer_ops.create_device_dense_output(self.runtime.host_buffers.dense),
            status=self.buffer_ops.create_device_status(self.runtime.host_buffers.status),
        )
        print("ensure_buffers")
        return


    def _check_shape(self, x: np.ndarray, expected: tuple[int, ...], name: str) -> None:
        if x.shape != expected:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {x.shape}")

    def _shape_check_all(self, R0, FREQ, PA, PS):
        ns = self._execution_spec.ns
        ups = self._execution_spec.ups
        tile = self._execution_spec.tile
        nk = self._model_spec.nk

        # Convert inputs to arrays first
        R0 = np.asarray(R0, dtype=np.float64)
        FREQ = np.asarray(FREQ, dtype=np.float64)
        PA = np.asarray(PA, dtype=np.float64)
        PS = np.asarray(PS, dtype=np.float64)

        # Shape consistency checks
        if R0.ndim != 2:
            raise ValueError(f"R0 must be 2D with shape ({ns}, {ups}) or ({ns}, {tile}), got {R0.shape}")
        if FREQ.ndim != 2:
            raise ValueError(f"FREQ must be 2D with shape ({nk}, {ns}), got {FREQ.shape}")
        if PA.ndim != 2:
            raise ValueError(f"PA must be 2D with shape ({nk}, {ns}), got {PA.shape}")
        if PS.ndim != 2:
            raise ValueError(f"PS must be 2D with shape ({nk}, {ns}), got {PS.shape}")

        if R0.shape not in ((ns, ups), (ns, tile)):
            raise ValueError(
                f"R0 shape mismatch: expected ({ns}, {ups}) or already padded ({ns}, {tile}), got {R0.shape}"
            )

        expected_k_ns = (nk, ns)
        if FREQ.shape != expected_k_ns:
            raise ValueError(f"FREQ shape mismatch: expected {expected_k_ns}, got {FREQ.shape}")
        if PA.shape != expected_k_ns:
            raise ValueError(f"PA shape mismatch: expected {expected_k_ns}, got {PA.shape}")
        if PS.shape != expected_k_ns:
            raise ValueError(f"PS shape mismatch: expected {expected_k_ns}, got {PS.shape}")
        

    def _make_solver(self, b: BackendName, v: KernelVariant) -> Callable[..., Any]:
        raise NotImplementedError



