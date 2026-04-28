"""
Bubble Position Control of N Coupled Bubbles in 1D Translational Motion

This module models and controls the spatial dynamics of N acoustically
coupled gas bubbles undergoing one-dimensional translational motion.

Key features:
- Supports an arbitrary number (N) of interacting bubbles
  (including small systems such as 2, 3, or 4 bubbles).
- Uses the CKM1D (Coupled Keller–Miksis 1D) model to describe
  bubble dynamics under acoustic excitation.
- Enables independent position control of each bubble by
  manipulating the external acoustic field.
- Accounts for inter-bubble coupling effects in both oscillatory
  and translational behavior.

Applications:
- ???
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List, Optional, Literal, Union, Tuple
from .common.specs import (
    SpaceSchema, 
    ObservationComponent,
    ActionComponent,
    ActionMapingManager,
    )

from .common.buffers import ObservationBufferContainer
from .common.interop import create_view
from .common.trajectory_buffer import TrajectorBuffer
from .bubble_vec_env import BubbleVecEnv


# MATHEMATICAL MODEL
from ..models import CKM1D
from ..models import MaterialProperties
from ..backends import BackendName, KernelVariant


class SceneGenerator:
    def __init__(
            self,
            num_bubbles: int,
            initial_position: Union[Literal["random", "equidistant"], List[float]],
            target_position: Union[Literal["random", "equidistant"], List[float]],
            initial_distance: Optional[float] = None,
            target_distance: Optional[float]  = None,
            alignment: Literal["center", "random"] = "center",
            min_distance: float = 0.05,
            max_distance: float = 0.15,
            margin: float = 0.05,
            domain_max: float = 1.0,
            domain_min: float = -1.0,
            seed: int = 42,
                 ) -> None:
        """
        Generate New Scenes for each episode
        """
        # --- Set Position Generation Method ---
        def get_position(position, name):
            if isinstance(position, str):
                if position.lower() in ["random", "equidistant"]:
                    return position
                else:
                    raise TypeError(f"Err: {name} must be 'random', 'equidistant' (str) or a sequence of numbers.")
            elif isinstance(position, list):
                if not len(position) == num_bubbles:
                    raise ValueError(f"Err: {name}length is {len(position)}, but {num_bubbles} is required.")
                return [float(p) for p in position]
            else:
                raise TypeError(f"Err: {name} must be 'random' (str) or a sequence of numbers.")

        self.target_position = get_position(target_position, "target_position")
        self.initial_position = get_position(initial_position, "initial_position") 

        # --- Attributes ---
        self.initial_distance = initial_distance
        self.target_distance  = target_distance
        self.alignment = alignment
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.margin = margin
        self.domain_max = domain_max
        self.domain_min = domain_min
        self.num_bubbles = num_bubbles

        # -- Random Generator --
        self._seed = seed
        self._generators = {}

    def _get_generator(self, device):
        device = torch.device(device)
        key = str(device)

        if key not in self._generators:
            gen = torch.Generator(device=device)

            if self._seed is not None:
                gen.manual_seed(self._seed)

            self._generators[key] = gen

        return self._generators[key]


    def generate_new_scenes(
            self,
            num_scenes,
            dtype: torch.dtype = torch.float32,
            device: str = "cpu",
            ) -> Tuple[torch.Tensor, ...]:
        
        # 1) Generate bubble positions
        if self.initial_position == "random":
            bubble_positions = self._random_uniform_position_1D(num_scenes, dtype, device)
        elif self.initial_position == "equidistant":
            bubble_positions = self._equidistant_position_1D(self.initial_distance, num_scenes, dtype, device)
        else:
            bubble_positions = self._expand_positions_1D(self.initial_position, num_scenes, dtype, device)
            self._random_uniform_position_1D(num_scenes, dtype, device)

        # 2) Generate new target positions
        if self.target_position == "random":
            target_positions = self._random_uniform_position_1D(num_scenes, dtype, device)
        elif self.target_position == "equidistant":
            target_positions = self._equidistant_position_1D(self.initial_distance, num_scenes, dtype, device)
        else:
            target_positions = self._expand_positions_1D(self.target_position, num_scenes, dtype, device)

        return bubble_positions, target_positions

    # -- Scene Generator Methods --
    def _equidistant_position_1D(self, distance, num_scenes, dtype, device):
        lo, hi = self.domain_min+self.margin, self.domain_max-self.margin
        if distance is None:
            # If no initial distance is provided, bubbles are distributed evenly across the domain.
            temp_pos = torch.linspace(lo, hi, self.num_bubbles, dtype=dtype, device=device)
            gap = temp_pos[1] - temp_pos[0]
            if gap < self.min_distance:
                raise ValueError(f"Cannot fit: The gaps between bubbles are {gap:.3f}, minimal gap is {self.min_distance:.2f} ")
            if gap > self.max_distance:
                raise ValueError(f"Cannot fit: The gaps between bubbles are {gap:.3f}, maximum gap is {self.max_distance:.2f} ")
            
            offset = torch.zeros((num_scenes, 1), dtype=dtype, device=device)

        elif type(distance) in [float, int]:
            # If an initial distance is specified, bubbles are placed with this fixed spacing between consecutive bubbles
            available_domain = hi - lo
            span_domain = (self.num_bubbles - 1) * distance
            if span_domain > available_domain:
                raise ValueError(f"Cannot fit: The scene requires min length of {span_domain:.2f}, available domain length is {available_domain:.2f}.")
            temp_pos = torch.arange(
                            self.num_bubbles, 
                            dtype=dtype,
                            device=device) * distance
            if self.alignment == "center":
                offset = torch.full(
                            (num_scenes, 1),
                            (lo + hi - span_domain) * 0.5,
                            dtype=dtype,
                            device=device)
            elif self.alignment == "random":
                max_start = hi - span_domain
                genenrator = self._get_generator(device)
                offset = lo + torch.rand(
                            (num_scenes, 1),
                            dtype=dtype,
                            device=device,
                            generator=genenrator
                        ) * (max_start - lo)
            else:
                raise ValueError("alignment must be: 'lower' | 'random'.")
                
        pos = temp_pos.unsqueeze(0) + offset
        return pos

    def _random_uniform_position_1D(self, num_scenes, dtype, device):
        lo, hi = self.domain_min+self.margin, self.domain_max-self.margin
        available_domain = hi - lo
        num_gaps = self.num_bubbles - 1
        min_required_domain = num_gaps * self.min_distance

        if min_required_domain > available_domain:
            raise ValueError(f"Cannot fit: The scene requires min length of {min_required_domain:.2f}, domain length is {available_domain:.2f}.")

        # Sample random extra spacing for each gap, then scale it if needed so that
        # all bubbles remain inside the available domain.
        # room \in [min_distance, max_distance]  --> dx_min < dx < dx_max
        room_per_gap = self.max_distance - self.min_distance
        extra_gap_limit = available_domain - min_required_domain    # Max offset
        generator = self._get_generator(device)

        # 1) extras_raw ~ U(0, room_per_gap)
        extras_raw = torch.rand(
                    (num_scenes, num_gaps), 
                    dtype=dtype,
                    device=device,
                    generator=generator) * room_per_gap
        
        # 2) Scale
        s = extras_raw.sum(dim=1)
        denom = torch.clamp(s, min=1e-12)
        scale = (extra_gap_limit / denom).clamp(max=1.0)
        extras = extras_raw * scale.unsqueeze(-1)

        # 3) Real gaps
        gaps = extras + self.min_distance
        remaining = available_domain - gaps.sum(dim=1)    # (num_scenes, )

        # 4) Choose the anchor position according to the requested alignment.
        if self.alignment == "center":
            offset = lo + (0.5 * remaining).unsqueeze(-1)
        elif self.alignment == "random":
            offset = lo + torch.rand(
                            (num_scenes, 1),
                            dtype=dtype,
                            device=device,
                            generator=generator
                    ) * remaining.unsqueeze(-1)
        else:
            raise ValueError("alignment must be: 'center' | 'random'.")
                
        # 5) Cumulative sums
        cumsum = torch.cumsum(gaps, dim=1)  # (num_envs, num_gaps)
        #print("cumsum\n", cumsum)
        #print("offset\n", offset)

        # 6) Concatenate the positions
        pos = torch.cat([offset, offset+cumsum], dim=1)

        #pos.clamp_(min=lo, max=hi)
        assert torch.all(pos >= lo - 1e-6)
        assert torch.all(pos <= hi + 1e-6)
        #print("pos\n", pos)

        return pos

    def _expand_positions_1D(self, position, num_scenes, dtype, device):
        positions = torch.tile(
                torch.as_tensor(position, dtype=dtype, device=device),
                (num_scenes, 1)
            )
        return positions


class PosNBC1D(BubbleVecEnv):
    """
    Pos - Position Control
    NB  - N-bubble system
    C   - Coupled bubbles
    1D  - 1 dimensional translation 
    """

    def __init__(
            self,
            num_bubbles: int,
            num_envs: int, 
            envs_per_block: int, 
            action_space_schema: Optional[SpaceSchema] = None, 
            observation_space_schema: Optional[SpaceSchema] = None,
            # Physical Parameters
            R0: Optional[List[float]] = None,
            PA: Optional[List[float]] = None,
            FR: Optional[List[float]] = None,
            PS: Optional[List[float]] = None,
            REF_FREQ: float = 25.0,        # kHz
            # Acoustic Field
            num_components: int = 2,
            acoustic_field: Literal["CONST", "SW_N", "SW_A"] = "SW_N",
            # Environment specifics
            episode_length: int = 128,                     # Number of RL-steps per episode
            time_step_length: Union[int, float] = 5.0,     # Timespan of integrations
            seed: int = 42,
            target_position: Union[Literal["random", "equidistant"], List[float]] = "random",
            initial_position: Union[Literal["random", "equidistant"], List[float]] = "random",
            initial_distance: Optional[float] = None,
            target_distance: Optional[float] = None,
            alignment: Literal["center", "random"] = "center",
            min_distance: float = 0.05,
            max_distance: float = 0.9,
            margin: float = 0.05,
            # Reward properties
            dnorm: float = 0.2,
            position_tolerance: float = 0.01,
            apply_termination: bool = True,
            rewards_weights: Optional[List[float]] = None,
            reward_exp: float = 0.5,
            positive_terminal_reward: float = 10.0,
            negative_terminal_reward: float = -10.0,
            # Render & Trajectory buffer properties
            render_env: bool = False,
            collect_trajectories: bool = True,
            trajectory_resolution: int = 128,
            trajectory_buffer_store: str = "PosNBC1D",
            # CUDA solver specifics
            backend: BackendName = BackendName.NUMBA,
            variant: KernelVariant = KernelVariant.WARP,
            max_kernel_steps: int = int(1e6),
            kernel_steps: int = 2048,
            # CUDA-OPTS
            compiler: Literal["nvrtc", "nvcc"] = "nvrtc",
            max_registers: Optional[int] = None,
            fastmath: bool = True,
            cuda_mode: Literal["debug", "profile", "release"] = "release",
            device: str = "cuda",
            ) -> None:
        
        # -- Action anb Observation space --
        if action_space_schema is None:
            self.action_space_schema = SpaceSchema()
            self.action_space_schema.components = [
                ActionComponent(
                    name = "PA",
                    idx  = [0, 1],
                    min  = [0.0, 0.0],
                    max  = [1.0, 1.0],
                    scale= 1.0e5,
                    ),
                ActionComponent(
                    name = "PS",
                    idx  = [0, 1],
                    min  = [0.0, 0.0],  
                    max  = [0.25, 0.25], # 0.5?,
                    scale= np.pi
                )
            ]
        else:
            self.action_space_schema = action_space_schema

        if observation_space_schema is None:
            self.observation_space_schema = SpaceSchema()
            self.observation_space_schema.components = [
                ObservationComponent(
                    name = "XT",
                    idx  = [i for i in range(num_bubbles)],
                    min  = [-0.5] * num_bubbles,
                    max  = [ 0.5] * num_bubbles,
                ),
                ObservationComponent(
                    name = "X",
                    idx  = [i for i in range(num_bubbles)],
                    min  = [-0.5] * num_bubbles,
                    max  = [ 0.5] * num_bubbles,
                    stack= 2
                )
            ]
        else:
            self.observation_space_schema = observation_space_schema
        
        # -- Initialize the base class --
        super().__init__(
            num_envs, envs_per_block, 
            self.action_space_schema, self.observation_space_schema,
            seed, 
            device)
        
        # -- Mathematical Model & GPU-solver configuration
        def init_list(x, size, name, default, scale: float = 1.0):
            src = x if x is not None else default
            if len(src) > size:
                raise ValueError(f"{name}: expected length {size}, got {len(src)}")
            lst = [float(v) * scale for v in src]
            return lst

        self.default_R0 = init_list(R0, num_bubbles,    "R0", [60.0] * num_bubbles)
        self.default_PA = init_list(PA, num_components, "PA", [1.0, 0.0])
        self.default_FR = init_list(FR, num_components, "FR", [25.0, 50.0])
        self.default_PS = init_list(PS, num_components, "PS", [0.0, 0.0])
        self.REF_FREQ   = REF_FREQ

        # -- Initializ with default values --
        R0_init = np.tile(
            np.array(self.default_R0, dtype=np.float64),
            (self.num_envs, 1))  # shape (num_systems, units_per_system), [micron]
        
        PA_init = np.tile(
            np.array(self.default_PA, dtype=np.float64)[:, np.newaxis],
            (1, self.num_envs)  # shape (num_components, num_systems), [bar]
        )

        FR_init = np.tile(
            np.array(self.default_FR, dtype=np.float64)[:, np.newaxis],
            (1, self.num_envs)  # shape (num_components, num_systems), [bar]
        )

        PS_init = np.tile(
            np.array(self.default_PS, dtype=np.float64)[:, np.newaxis],
            (1, self.num_envs)  # shape (num_components, num_systems), [bar]
        )

        self.model = CKM1D(
            num_systems = self.num_envs,
            systems_per_block = self.envs_per_block,
            units_per_system  = num_bubbles,
            num_components = num_components,
            acoustic_field = acoustic_field,
            num_stored_points = trajectory_resolution if collect_trajectories else 0,
            # Phyisical Properties
            R0 = R0_init,
            PA = PA_init,
            FR = FR_init,
            PS = PS_init,
            REF_FREQ = self.REF_FREQ,
            # BACKED OPTS
            backend = backend,
            variant = variant,
            # CUDA-OPTS
            cuda_mode = cuda_mode,
            compiler = compiler,
            max_registers = max_registers,
            fastmath = fastmath
        )

        self.solver_kwargs = {
                "debug"            : False,
                "max_steps"        : max_kernel_steps,
                "kernel_steps"     : kernel_steps
            }
        
        # -- Action Map & Observation Buffer --
        self._action_map = ActionMapingManager(self.action_space_schema,
                                               self.model.control_layout)
        
        self._observation_buffers = ObservationBufferContainer(
                                    self.num_envs,
                                    self.observation_space_schema,
                                    dtype=torch.float32,
                                    device=device)

        # -- Environment specific attributes --
        self.episode_length     = episode_length
        self.time_step_length   = time_step_length
        self.seed               = seed
        self.num_components     = num_components
        self.num_bubbles        = num_bubbles
        self.device             = device

        # -- Scene generator  --
        # Get Domain limits from observation
        self.domain_max = max(max(comp.max) for comp in self.observation_space_schema.components if comp.name in ["X", "XT"])
        self.domain_min = min(min(comp.min) for comp in self.observation_space_schema.components if comp.name in ["X", "XT"])

        self.scene_generator = SceneGenerator(
            self.num_bubbles,
            initial_position,
            target_position,
            initial_distance,
            target_distance,
            alignment,
            min_distance,
            max_distance,
            margin,
            self.domain_max,
            self.domain_min,
            seed = self.seed)

        # -- Reward function params
        self.reward_params = {
            "weights" :  torch.tensor(init_list(rewards_weights, 2, "reward_weights", [1.0, 0.0]), dtype=torch.float32, device=device),
            "reward_exp" : reward_exp,
            "r_dnorm"    : 1.0 / dnorm,
            "positive_terminal_reward" : positive_terminal_reward,
            "negative_terminal_reward" : negative_terminal_reward,
        }
        self.apply_termination = apply_termination
        self.position_tolerance = position_tolerance
        
        # -- Render & Trajector buffer properties
        self._render_env = render_env
        self._collect_trajectories = collect_trajectories
        # TODO: general metadata!!
        self.trajectory_buffer = TrajectorBuffer(
            num_envs=self.num_envs,
            num_units=self.num_bubbles,
            saved_states=[0, 1],
            zarr_root=trajectory_buffer_store,
            metadata={
                "num_bubbles"   : self.num_bubbles,
                "ac_type"       : acoustic_field,
                "k"             : num_components,
                "PA_max"        : self.action_space_schema.components[0].max,
                "PA_min"        : self.action_space_schema.components[0].min,
                "PS_max"        : self.action_space_schema.components[1].max,
                "PS_min"        : self.action_space_schema.components[1].max,
                "freq"          : self.default_FR,
                "ref_freq"      : self.REF_FREQ,
                "XT_max"        : self.observation_space_schema.components[0].max,
                "XT_min"        : self.observation_space_schema.components[0].min,
                "X_max"         : self.observation_space_schema.components[1].max,
                "X_min"         : self.observation_space_schema.components[1].max,
                "position_tolerance" : self.position_tolerance,
                "time_step_length" : self.time_step_length
            }
        ) if self._collect_trajectories else None

    @torch.no_grad()
    def step(self, action: Optional[torch.Tensor] = None, clone: bool = False):
        # DEBUG RANDOM ACTIONS --> SMOKE, delete later
        if action is None:
            action = self.action_space.sample()
        self.set_action(action)
        self.step_physics()
        self.observe()
        self.check_termination()
        self.get_rewards()
        self.advance_time()
        if self.trajectory_buffer is not None:
            self.trajectory_buffer.step(
                self._observations,
                self._actions,
                self._rewards,
                self._interop["dense_index"],
                self._interop["dense_time"],
                self._interop["dense_state"]
            )
            
        if self._render_env:
            self.render()
        info = self.final_termination()

        if clone:
            return (self._observations.clone(),
                    self._rewards.clone(),
                    self._terminateds.clone(),
                    self._time_outs.clone(),
                    info)
        else:
            return (self._observations,
                    self._rewards,
                    self._terminateds,
                    self._time_outs,
                    info)


    @torch.no_grad()
    def reset(self, **kwargs):
        """ Reset Host Buffers State and initialize buffers & counters 
        Note: the device buffers copy host buffer values on initialization, explicit copy not required """
        bubble_positions, \
        target_positions   \
            = self.scene_generator.generate_new_scenes(
                num_scenes=self.num_envs, 
                dtype=torch.float64,
                device="cpu")
        
        # -- Initialize Host Buffers --
        self.model.runtime.host_buffers.state.actual_state[self.model.state_layout["R"].base_offset, :, :self.num_bubbles] = 1.0   # Radius
        self.model.runtime.host_buffers.state.actual_state[self.model.state_layout["X"].base_offset, :, :self.num_bubbles] = bubble_positions.numpy()
        self.model.runtime.host_buffers.state.actual_state[self.model.state_layout["U"].base_offset, :, :self.num_bubbles] = 0.0   # Wall Velocity
        self.model.runtime.host_buffers.state.actual_state[self.model.state_layout["V"].base_offset, :, :self.num_bubbles] = 0.0   # Translational Velocity
        self.model.runtime.host_buffers.state.time_end[:] = self.time_step_length
        self.model._ensure_buffers()    # Call to create device buffers

        # -- Update Observation Buffers --
        # TODO: adaptive buffer!
        self._observation_buffers.update_buffer("XT", target_positions)
        self._observation_buffers.update_buffer("X", bubble_positions, fill=True)
        #print("-- Observation --")
        self._get_observation()
        #print(self._observation_buffers.get_stacked())

        # -- Initialize Metrics and Counters --
        self.algo_steps     = torch.zeros(size=(self.num_envs, ), dtype=torch.long, device=self.device)
        self.total_rewards  = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device=self.device)

        # Build interops
        self._build_interop_views()

        # Info
        info = {}

        # TODO: clone or not clone?
        return self._observations.clone(), info


    @torch.no_grad()
    def reset_envs(self, env_ids, sync_host: bool = False, **kwargs):
        """
        Reset Terminated envs and counters. Device values are set via interop
        """
        #print("reset-us-please\n", env_ids)
        #input()
        bubble_positions, \
        target_positions \
            = self.scene_generator.generate_new_scenes(
                num_scenes=env_ids.numel(),
                dtype=torch.float64,
                device=self.device)
        
        # Interop
        self._interop["actual_state"][self.model.state_layout["R"].base_offset][:, :self.num_bubbles].index_fill_(0, env_ids, 1.0)
        self._interop["actual_state"][self.model.state_layout["X"].base_offset][:, :self.num_bubbles].index_copy_(0, env_ids, bubble_positions)
        self._interop["actual_state"][self.model.state_layout["U"].base_offset][:, :self.num_bubbles].index_fill_(0, env_ids, 0.0)
        self._interop["actual_state"][self.model.state_layout["V"].base_offset][:, :self.num_bubbles].index_fill_(0, env_ids, 0.0)

        # -- Upate Observation Buffers --
        # TODO: adaptive buffer!
        self._observation_buffers.update_buffer_at("XT", env_ids, target_positions)
        self._observation_buffers.update_buffer_at("X", env_ids, bubble_positions, fill=True)
        self._get_observation()

        # -- Reset Metrics and Counters --
        self.algo_steps[env_ids] = 0
        self.total_rewards[env_ids] = 0.0

        if sync_host:
            env_ids_np = env_ids.cpu().numpy()
            self.model.runtime.host_buffers.state.actual_state[self.model.state_layout["X"], env_ids_np, :self.num_bubbles] = bubble_positions.cpu().numpy()

    @torch.no_grad()
    def close(self):
        print("close")


    @property
    @torch.no_grad()
    def target_positions(self):
        return self._observation_buffers.get("XT")

    @property
    @torch.no_grad()
    def bubble_positions(self):
        return self._observation_buffers.get("X")

    @property
    @torch.no_grad()
    def target_distance(self):
        return torch.abs(self.target_positions - self.bubble_positions)


    @torch.no_grad()
    def render(self):
        plot_max_envs = min(16, self.num_envs)
        target_positions = self.target_positions[:plot_max_envs, :].cpu().numpy()
        bubble_positions = self.bubble_positions[:plot_max_envs, :].cpu().numpy()

        actions = getattr(self, "_actions", None)
        if actions is not None:
            actions = actions[:plot_max_envs].cpu().numpy()

        rewards = getattr(self, "_rewards", None)
        if rewards is not None:
            rewards = rewards[:plot_max_envs].cpu().numpy()

        x = getattr(self, "_render_x", None)
        if x is None:
            x = np.arange(plot_max_envs, dtype=np.int32)
            self._render_x = x

        x_rep = np.repeat(x, self.num_bubbles)      # (plot_max_envs x num_bubbles, )
        y_t_flat = target_positions.reshape(-1)     # (plot_max_envs x num_bubbles, )
        y_b_flat = bubble_positions.reshape(-1)     # (plot_max_envs x num_bubbles, )

        if not hasattr(self, "_render_init") or not self._render_init:
            plt.ion()
            self.fig = plt.figure(figsize=(7, 8))
            gs  = self.fig.add_gridspec(3, 1, hspace=0.35)
            ax0 = self.fig.add_subplot(gs[0, 0])
            ax1 = self.fig.add_subplot(gs[1, 0], sharex=ax0)
            ax2 = self.fig.add_subplot(gs[2, 0], sharex=ax0)
            self._axes = (ax0, ax1, ax2)

            # ax0: target (+) and bubble (.) positions
            self._h_target = ax0.scatter(x_rep, y_t_flat, marker="+", s=20, linewidths=1.2, label="Target")
            self._h_bubble = ax0.scatter(x_rep, y_b_flat, marker=".", s=15, label="Bubble")
            ax0.set_ylabel(r"Positions ($x / \lambda$)")
            pad = 0.1 * (self.domain_max - self.domain_min + 1e-9)
            ax0.set_ylim(self.domain_min - pad, self.domain_max + pad)

            if actions is not None:
                self._h_actions = []
                for i in range(actions.shape[-1]):
                    self._h_actions.append(ax1.scatter(x, actions[:,i], marker=".", s=15))
                ax1.set_ylabel("actions")
            ax1.set_ylim(0, 1.2)

            if rewards is not None:
                self._h_rewards = ax2.scatter(x, rewards, marker=".", s=20)
                ax2.set_ylabel("rewards")
                ax2.set_ylim(-1, 1)

            self._render_init = True
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return
        
        # -- Update plots --
        self._h_target.set_offsets(np.column_stack([x_rep, y_t_flat]))
        self._h_bubble.set_offsets(np.column_stack([x_rep, y_b_flat]))

        if actions is not None:
            for i in range(actions.shape[-1]):
                self._h_actions[i].set_offsets(np.column_stack([x, actions[:, i]]))

        if rewards is not None:
            self._h_rewards.set_offsets(np.column_stack([x, rewards]))

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


    def _build_interop_views(self):
        db = self.model.runtime.device_buffers
        self._interop = {
            "time_step"         : create_view(db.state.time_step, "time_step"),
            "actual_time"       : create_view(db.state.actual_time, "actual_time"),
            "actual_state"      : create_view(db.state.actual_state, "actual_state"),
            "control_params"    : create_view(db.params.control_params, "control_params"),
            "status_flags"      : create_view(db.status.status_flags, "status_flags"),
            "actual_event"      : create_view(db.status.actual_event, "actual_event")
        }

        if self._collect_trajectories:
            self._interop["dense_state"] = create_view(db.dense.dense_state, "dense_state")
            self._interop["dense_time"]  = create_view(db.dense.dense_time, "dense_time")
            self._interop["dense_index"] = create_view(db.dense.dense_index, "dense_index")

    @torch.no_grad()
    def _get_observation(self):
        self._observations = self.observation_space.scale(
            self._observation_buffers.get_stacked())


    # -- STEP SUB-METHODS --
    @torch.no_grad()
    def set_action(self, actions: torch.Tensor, ptr: Optional[int] = None, size: Optional[int] = None):
        # pytorch interop!
        tbuf = self._interop["control_params"].view(self.num_components * 4, self.num_envs)
        self._actions = actions
        a64 = actions.to(device=tbuf.device, dtype=torch.float64)
        scales = getattr(self, "_action_scales", None)
        if scales is None:
            scales = torch.as_tensor(self._action_map._scales, dtype=torch.float64, device=a64.device)
            self._action_scales = scales
        dev_idx = getattr(self, "_action_dev_idx", None)
        if dev_idx is None:
            dev_idx = torch.as_tensor(self._action_map._dev_cols, dtype=torch.long, device=tbuf.device)
            self._action_dev_idx = dev_idx

        a64.mul_(scales)
        a64 = a64.transpose(0, 1)
        #a64.transpose_(0, 1)
        tbuf.index_copy_(0, dev_idx, a64)

    def step_physics(self):
        """ Wrap model.solve to provide an API call"""
        self.model.solve(**self.solver_kwargs)

    @torch.no_grad()
    def observe(self):
        for comp in self.observation_space_schema.components:
            name = comp.name
            entry = self.model.state_layout.get(comp.name)
            if entry is None:
                continue
            if entry.group not in self._interop:
                raise KeyError(f"Missing interop buffer for group '{entry.group}'")
            full_state = self._interop[entry.group]
            tbuf = full_state[entry.base_offset, :, :self.num_bubbles]
            tbuf32 = tbuf.to(dtype=torch.float32)
            self._observation_buffers.update_buffer(name, tbuf32)

        self._get_observation()

    @torch.no_grad()
    def check_termination(self):
        # 1) Check and Reset Status
        self._failure = (self._interop["status_flags"] != 1).squeeze()
        self._interop["status_flags"].zero_()

        # 2) Episode success
        if self.apply_termination:
            self._success = torch.all(self.target_distance <= self.position_tolerance, dim=1)
        else:
            self._success = torch.zeros_like(self._failure, dtype=torch.bool)

        if torch.any(self._failure):
            print("Number of episode failures:", self._failure.sum().item())

        if torch.any(self._success):
            print("Number of episode success:", self._success.sum().item())

        self._terminateds = self._failure | self._success

    @torch.no_grad()
    def get_rewards(self):
        # Mask target distance
        td = self.target_distance
        td = td.masked_fill(td < self.position_tolerance, 0.0)

        # Distance Penalty
        norm_dist = (td * self.reward_params["r_dnorm"]).clamp(min=0.0)
        distance_term = torch.mean(norm_dist.pow(self.reward_params["reward_exp"]), dim=1)

        # Actuation Penalty
        pa_idx = self._action_map._feat_col["PA"]
        intensity = torch.sum(self._actions[:, pa_idx].pow(2.0), dim=1).sqrt()

        # Weighted rewards
        self._rewards = (
            - self.reward_params["weights"][0] * distance_term
            - self.reward_params["weights"][1] * intensity
            + self.reward_params["negative_terminal_reward"] * self._failure.float()
            + self.reward_params["positive_terminal_reward"] * self._success.float()
        )

        self.total_rewards += self._rewards

    @torch.no_grad()
    def advance_time(self):
        self._interop["actual_time"].zero_()
        self._interop["time_step"].fill_(1e-6)
        self._interop["actual_event"].fill_(-1.0)
        self.algo_steps += 1

        self._time_outs = self.algo_steps >= self.episode_length
        # Note real time is algo_steps x time_step_length + time

        if torch.any(self._time_outs):
            print("Number of episode time-outs:", self._time_outs.sum().item())


    @torch.no_grad()
    def final_termination(self):
        info = {}
        done_env_idx = torch.nonzero(self._time_outs | self._terminateds).flatten()
        num_done_envs = done_env_idx.numel()
        if num_done_envs > 0:
            info = {
                "final_observation" : self._observations[done_env_idx].clone(),
                "dones"             : done_env_idx,
                "episode_return"    : self.total_rewards[done_env_idx].clone(),
                "episode_length"    : self.algo_steps[done_env_idx].clone()
            }

            if self.trajectory_buffer is not None:
                self.trajectory_buffer.end_episode(
                    done_env_idx,
                    info["final_observation"],
                    info["episode_length"],
                    info["episode_return"],
                    np.tile(self.default_R0, (num_done_envs, 1))
                )

            self.reset_envs(done_env_idx)

        return info

