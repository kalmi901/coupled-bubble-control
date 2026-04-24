"""
Coupled Keller--Miksis Bubble Model with 1D Translational motion
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Callable, Any
from .model import (
    Model,
    MaterialProperties,
    ModelRuntime
)
from .specs import (
    ModelSpec,
    ExecutionSpec,
    SolverSpec,
    LayoutEntry)

from .buffers import (
    BufferSet,
    ActualStateBuffer,
    ParameterBuffer,
    DenseOutputBuffer,
    BufferOps)

from ..backends import BackendName, KernelVariant, CUDAOpts


class CKM1D(Model):
    _available_backends = (BackendName.NUMBA, BackendName.CUPY)
    _kernel_variants = (KernelVariant.SHARED, )
    def __init__(
            self,
            num_systems: int,
            systems_per_block: int,
            units_per_system: int,
            # Physical parameters
            R0: NDArray[np.float64],  # shape: (num_systems, units_per_system), [micron]
            PA: NDArray[np.float64],  # shape: (num_componens, num_systems),    [bar]
            FR: NDArray[np.float64],  # shape: (num_componens, num_systems),    [kHz]
            PS: NDArray[np.float64],  # shape: (num_componens, num_systems),    [-]
            REF_FREQ: float = 20.0,   # kHz
            # Acoustuc field
            num_components: int = 2,
            acoustic_field: Literal["CONST", "SW_N", "SW_A"] = "CONST",
            mat_props: Optional[MaterialProperties] = None,
            # Solver specifics
            backend: BackendName = BackendName.NUMBA,
            variant: KernelVariant = KernelVariant.SHARED,
            num_stored_points: int = 0,
            # CUDA-OPTS:
            compiler: Literal["nvrtc", "nvcc"] = "nvcc",
            max_registers: Optional[int] = None,
            fastmath: bool = True,
            cuda_mode: Literal["debug", "profile", "release"] = "release"
            ) -> None:
        
        # INITIALIZE SPECIFICATIONS
        self._model_spec = ModelSpec(
            name = "CKM1D",
            sd  = 4,                # System Dimension (R, X, U, V)
            nup = 13,               # Number of unit parameters (specific for each unit)
            nsp = 5,                # Number of system parameters (shared by all units)      
            ncm = 3,                # Number of coupling matrices (R-T, T-R, T-T)
            ne  = 1,                # Number of events (bubble collision)
            ncf = 4,                # Number of coupling factors
            nct = 8,                # Number of couplint terms
            nk = num_components,
            ac = acoustic_field
        )

        self._execution_spec = ExecutionSpec(
            ns = num_systems,
            spb = systems_per_block,
            ups = units_per_system,
            ndo = num_stored_points
        )

        self._cuda_opts = CUDAOpts(
            mode=cuda_mode,
            fastmath = fastmath,
            max_registerts=max_registers,
            compiler = compiler
        )

        self._solver_spec = SolverSpec()
        self.runtime = ModelRuntime(backend, variant)
        self.buffer_ops = BufferOps(backend)

        self.control_layout = {
            "PA" : LayoutEntry("control_params", 0),
            "FR" : LayoutEntry("control_params", num_components),
            "PS" : LayoutEntry("control_params", 2 * num_components),
            "WL" : LayoutEntry("control_params", 3 * num_components)
        }

        self.state_layout = {
            "R" : LayoutEntry("actual_state", 0),
            "X" : LayoutEntry("actual_state", 1),
            "U" : LayoutEntry("actual_state", 2),
            "V" : LayoutEntry("actual_state", 3)
        }

        # INITIALIZE MATERIAL PROPERTIES
        if mat_props is not None:
            self.mat_props = mat_props
        else:
            self.mat_props = MaterialProperties()
        
        # INITIALIZE PHYSICAL PARAMTERS 
        self._shape_check_all(R0, FR, PA, PS)
        
        # Convert to SI units
        self._REF_FREQ = REF_FREQ * 1000.0                      # kHz --> Hz
        self._FREQ = np.ascontiguousarray(FR, dtype=np.float64) * 1000.0        # kHz --> Hz
        self._PA = np.ascontiguousarray(PA, dtype=np.float64) * 1.0e5             # bar --> Pa
        self._PS = np.ascontiguousarray(PS, dtype=np.float64) * np.pi             # unit--> rad

        # R0 handling with safe padding
        if R0.shape == (self._execution_spec.ns, self._execution_spec.tile):
            print("NO Padding Applied (already padded)")
            self._R0 = np.ascontiguousarray(R0, dtype=np.float64) * 1.0e-6        # um --> m
        else:
            print("Padding Applied")
            self._R0 = np.zeros((self._execution_spec.ns, self._execution_spec.tile), dtype=np.float64)
            self._R0[:, :self._execution_spec.ups] = R0 * 1e-6                   # um -> m
            self._R0 = np.ascontiguousarray(self._R0)

        # INITIALIZE PARAMETERS
        self._initialize_host_buffers()


    def build_parameters(self):
        unit_params = np.zeros(self.unit_param_shape, dtype=np.float64)
        system_params = np.zeros(self.system_param_shape, dtype=np.float64)
        coupling_matrices = np.zeros(self.coupling_matrix_shape, dtype=np.float64)

        # Refrence properties
        TWO_PI = 2.0 * np.pi
        PV  = self.mat_props.PV
        RHO = self.mat_props.RHO
        ST  = self.mat_props.ST
        VIS = self.mat_props.VIS
        CL  = self.mat_props.CL
        P0  = self.mat_props.P0
        PE  = self.mat_props.PE
        UPS = self._execution_spec.ups 

        w_ref = TWO_PI * self._REF_FREQ         # ω_ref
        l_ref = CL / self._REF_FREQ             # λ_ref
        R0_real = self._R0[:,:UPS]

        # Keller--Miksis
        unit_params[0,:,:UPS] = (2.0 * ST / R0_real + P0 - PV) * (TWO_PI / R0_real / w_ref)**2.0 / RHO
        unit_params[1,:,:UPS] = (1.0 - 3.0*PE) * (2 * ST / R0_real + P0 - PV) * (2.0*np.pi / R0_real / w_ref) / CL / RHO
        unit_params[2,:,:UPS] = (P0 - PV) * (2.0 *np.pi / R0_real / w_ref)**2.0 / RHO
        unit_params[3,:,:UPS] = (2.0 * ST / R0_real / RHO) * (TWO_PI / R0_real / w_ref)**2.0
        unit_params[4,:,:UPS] = 4.0 * VIS / RHO / (R0_real**2.0) * (2.0* np.pi / w_ref)
        unit_params[5,:,:UPS] = ((2.0 * np.pi / R0_real / w_ref)**2.0) / RHO
        unit_params[6,:,:UPS] = ((2.0 * np.pi / w_ref)** 2.0) / CL / RHO / R0_real
        unit_params[7,:,:UPS] = R0_real * w_ref / (2 * np.pi) / CL

        # Translational motion
        unit_params[8,:,:UPS]  = (0.5 * l_ref / R0_real)**2
        unit_params[9,:,:UPS]  = TWO_PI / RHO / R0_real / l_ref / (w_ref * R0_real)**2.0
        unit_params[10,:,:UPS] = 2 * TWO_PI / 3 * R0_real**3
        unit_params[11,:,:UPS] = 6 * TWO_PI * VIS * R0_real
        unit_params[12,:,:UPS] = R0_real / l_ref 

        # Shared parameters
        system_params[0] = 3.0 * PE
        system_params[1] = 1.0 / w_ref         # 1/ω_ref
        system_params[2] = l_ref / TWO_PI
        system_params[3] = CL
        system_params[4] = 1 / RHO / CL

        # Coupling matrices
        l_ref_mx = np.full_like(R0_real, l_ref)
        for row in range(self._execution_spec.ups):
            coupling_matrices[0,row,:,:UPS] = R0_real[:,row:row+1]**3 / R0_real **2 / l_ref
            coupling_matrices[1,row,:,:UPS] = 3.0 * (R0_real[:,row:row+1] / l_ref_mx)**3
            coupling_matrices[2,row,:,:UPS] = (18 * VIS / RHO) * (TWO_PI / w_ref) * (R0_real[:,row:row+1] / l_ref)**3 / (R0_real **2)

        # --> Disable self coupling
        idx = np.arange(self._execution_spec.ups)
        coupling_matrices[:, idx, :, idx] = 0.0

        return unit_params, system_params, coupling_matrices

    def build_control_parameters(self):
        control_params = np.zeros(self.control_param_shape, dtype=np.float64)
        TWO_PI = 2.0 * np.pi
        CL = self.mat_props.CL

        # Acoustic parameters
        NC = self._model_spec.nk
        control_params[0:NC, :]     = self._PA                      # Pressure Amplitude (Pa)
        control_params[NC:2*NC, :]  = self._FREQ * TWO_PI           # Angulare Frequency (ω_i, rad/s)
        control_params[2*NC:3*NC,:] = self._PS                      # Phase shift (rad)
        control_params[3*NC:4*NC,:] = TWO_PI * self._FREQ / CL      # Wave number  

        return control_params

    def build_default_initial_state(self):
        state = np.zeros(self.state_shape, dtype=np.float64)
        ups = self._execution_spec.ups
        state[0,:,:ups] = 1.0          # Dimensionless bubble radius
        state[1,:,:ups] = 0.0          # Dimensionless bubble position (correct later!)
        state[2,:,:ups] = 0.0          # Dimensionless wall velocity
        state[3,:,:ups] = 0.0          # Dimensionless tranalsationl velocity

        return state

    def _make_solver(self, b: BackendName, v: KernelVariant = KernelVariant.SHARED) -> Callable[..., Any]:
        from functools import partial
        if b in self._available_backends and b == BackendName.NUMBA:
            from ..backends.numba.rkck45_coupled_numba_solver import solver
            if v == KernelVariant.SHARED:
                from ..backends.numba.sysmtem_definitions.ckm1d_def import make_model
                from ..backends.numba.rkck45_coupled_numba_stepper import make_kernel
            elif v == KernelVariant.WARP:
                from ..backends.numba.sysmtem_definitions.ckm1d_def_warp_sync import make_model
                from ..backends.numba.rkck45_coupled_numba_stepper_warp_sync import make_kernel
            else:
                pass
            #from ..backends.numba.sysmtem_definitions.ckm1d_def_warp_sync import make_model
            ode_fun, event_fun = make_model(self._model_spec, self._execution_spec, self._solver_spec,
                                            self._cuda_opts)
            kernel = make_kernel(ode_fun, event_fun,
                    self._model_spec, self._execution_spec, self._solver_spec,
                    self._cuda_opts)
            return partial(solver, kernel=kernel)
        elif b in self._available_backends and b == BackendName.CUPY:
            from ..backends.cupy_cuda.kernel_loader import load_coupled_kernel
            from ..backends.cupy_cuda.rkck45_coupled_cupy_solver import solver
            if v == KernelVariant.SHARED:
                solver_src = "rkck45_coupled_cuda_sover.cu"
                system_src = "ckm1d_def.cuh"
            elif v == KernelVariant.WARP:
                solver_src = "rkck45_coupled_cuda_sover_warp_sync.cu"
                system_src = "ckm1d_def_warp_sync.cuh"
            kernel = load_coupled_kernel(
                solver_src,
                system_src,
                "rkck45_coupled_solver",
                self._model_spec,
                self._execution_spec,
                self._solver_spec,
                self._cuda_opts
            )
            return partial(solver, kernel=kernel)
        else:
            raise NotImplementedError


