from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union

from coupledbubble_control.envs.common.specs import (
    ActionComponent, ObservationComponent, SpaceSchema)

__all__ = [
    "Experiment",
    "StaticEnvFeatures",
    "DynamicEnvFeatures",
    "ActionSpace",
    "ObservationSpace",
    "CUDAOpts",
]

@dataclass
class CUDAOpts:
    backend: Literal["numba", "cupy"] = "numba"
    variant: Literal["shared", "warp"] = "warp"
    cuda_mode: Literal["release"] = "release"
    max_registers: int = 192

@dataclass
class Experiment:
    config_file: Optional[str] = None
    project_name: str = "CoupledBubblesPositionControl"
    trial_name: str = "PA_PS_Control"
    total_timesteps: int = int(5e6)
    eval_episodes: int = int(1e4)
    num_saved_trajectories: int = int(5e3)
    log_frequency: int = 1
    seed: int = 42
    log_training: bool = False
    render_env: bool = False


@dataclass
class StaticEnvFeatures:
    """Static Environment Features """
    number_of_bubbles: int = 2
    equilibrium_radius: List[float] = field(default_factory = lambda: [60.0])
    """microns"""
    time_step_length: int = 5
    max_steps_per_episode: int = 128
    initial_position: Union[List[float], Literal["random"]]= "random"
    target_position: Union[List[float], Literal["random"]] = "random"
    initial_distance: Union[float, None] = None
    final_distance: Union[float, None] = None
    alignment: Literal['center', 'random'] = "center"
    distance_range: List[float] = field(default_factory = lambda: [0.1, 0.9])
    dnorm: float = 0.2
    apply_termination: bool = True
    position_tolerance: float = 0.01
    positive_terminal_reward: float = 50.0
    negative_terminal_reward: float = -100.0
    reward_weights: List[float] = field(default_factory = lambda: [1.0, 0.0])
    reward_shape_exp: float = 0.5

    def __post_init__(self):
        n = int(self.number_of_bubbles)
        if n < 2:
            raise ValueError(f"numbef_of_bubbles must be >=2 (got {n})")
        
        # Broadcast
        if len(self.equilibrium_radius) == 1 and n >=2:
            self.equilibrium_radius = self.equilibrium_radius * n
        
        elif len(self.equilibrium_radius) != n:
            raise ValueError(
                f"equilibrium_radius length ({len(self.equilibrium_radius)}) must equal number_of_bubbles ({n})"
            )
        
        if len(self.distance_range) !=2:
            raise ValueError(
                f"len(distance_renge) must be 2 (got {len(self.distance_range)})"
            )

@dataclass
class DynamicEnvFeatures:
    """Dynamic Environment Features.
    RL-agent tune the pressure_amps and phase_sift values
    """
    number_of_harmonics: int = 2
    acoustic_field_type: Literal["SW_N", "SW_A", "CONST"] = "SW_N"
    excitation_freqs: List[float] = field(default_factory= lambda: [25.0, 50.0])  # kHz
    """kHz"""
    phase_shift: List[float] = field(default_factory=lambda: [0.0])             # rad
    """radians"""
    pressure_amps: List[float] = field(default_factory=lambda: [0.0])           # bar
    """bar"""
    def __post_init__(self):
        # Koerció tuple-re (tyro listát is adhat)
        self.excitation_freqs = self.excitation_freqs
        self.phase_shift     = self.phase_shift
        self.pressure_amps   = self.pressure_amps

        n = int(self.number_of_harmonics)
        if n < 1:
            raise ValueError(f"number_of_harmonics must be >= 1 (got {n})")
        self.number_of_harmonics = n

        # Frekvenciák: hossz és érvényesség
        if len(self.excitation_freqs) != n:
            raise ValueError(
                f"excitation_freqs length ({len(self.excitation_freqs)}) must equal number_of_harmonics ({n})"
            )

        # Broadcast phase/amp, ha 1 elem és n>1
        if len(self.phase_shift) == 1 and n > 1:
            self.phase_shift = self.phase_shift * n
        if len(self.pressure_amps) == 1 and n > 1:
            self.pressure_amps = self.pressure_amps * n

        # Hosszok egyezése
        if len(self.phase_shift) != n:
            raise ValueError(
                f"phase_shift length ({len(self.phase_shift)}) must equal number_of_harmonics ({n})"
            )
        if len(self.pressure_amps) != n:
            raise ValueError(
                f"pressure_amps length ({len(self.pressure_amps)}) must equal number_of_harmonics ({n})"
            )

@dataclass
class ObservationSpace:
    observe_target_position: bool = True
    observe_bubble_position: bool = True
    # Limits
    target_observation_minimum: float = -0.5
    target_observation_maximum: float =  0.5
    bubble_observation_minimum: float = -0.5
    bubble_observation_maximum: float =  0.5
    stacked_positions: int = 2

    def __post__init__(self):
        if not self.observe_target_position and not self.observe_bubble_position:
            raise ValueError("Enable at least one observable variable (X/XT).")
        
        if self.observe_target_position:
            if not (self.target_observation_minimum < self.target_observation_maximum):
                raise ValueError("amplitude_minimum < amplitude_maximum")
        if self.observe_bubble_position:
            if not (self.bubble_observation_minimum < self.bubble_observation_maximum):
                raise ValueError("phase_minimum < phase_maximum.")
            
    def get_space(self, num_bubbles: int):
        if num_bubbles < 1:
            raise ValueError("n must be >= 1")
        
        d = SpaceSchema()
        d.components = []
        if self.observe_target_position:
            d.components.append(
                ObservationComponent(
                    name = "XT",
                    idx  = list(range(num_bubbles)),
                    min  = [self.target_observation_minimum] * num_bubbles,
                    max  = [self.target_observation_maximum] * num_bubbles
            ))
        if self.observe_bubble_position:
            d.components.append(
                ObservationComponent(
                    name = "X",
                    idx  = list(range(num_bubbles)),
                    min  = [self.bubble_observation_minimum] * num_bubbles,
                    max  = [self.bubble_observation_maximum] * num_bubbles,
                    stack=self.stacked_positions 
                )
            )
        
        return d


@dataclass
class ActionSpace:
    control_amplitude: bool = True
    amplitude_minimum: float = 0.0
    amplitude_maximum: float = 1.0

    control_phase: bool = True
    phase_minimum: float = 0.0
    phase_maximum: float = 1.0

    def __post_init__(self):
        if not self.control_amplitude and not self.control_phase:
            raise ValueError("Enable at least one action component (PA/PS).")

        if self.control_amplitude:
            if not (self.amplitude_minimum < self.amplitude_maximum):
                raise ValueError("amplitude_minimum < amplitude_maximum.")
        if self.control_phase:
            if not (self.phase_minimum < self.phase_maximum):
                raise ValueError("phase_minimum < phase_maximum.")
            
    
    def get_space(self, num_components: int):
        if num_components < 1:
            raise ValueError("n must be >= 1")
        
        d = SpaceSchema()
        d.components = []
        if self.control_amplitude:
            d.components.append(
                ActionComponent(
                    name = "PA",
                    idx  = list(range(num_components)),
                    min  = [self.amplitude_minimum] * num_components,
                    max  = [self.amplitude_maximum] * num_components,
                    scale= 1.0e5
                )
            )
        if self.control_phase:
            from numpy import pi
            d.components.append(
                ActionComponent(
                    name = "PS",
                    idx  = list(range(num_components)),
                    min  = [self.phase_minimum] * num_components,
                    max  = [self.phase_maximum] * num_components,
                    scale= pi
                )
            )
        return d