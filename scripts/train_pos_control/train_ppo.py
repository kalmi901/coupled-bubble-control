from __future__ import annotations
import train_bootstrap
import yaml
import tyro
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import LiteralString, List, Dict, Annotated, Optional, Sequence, Literal, Any, Union
from typing import get_type_hints, get_origin, get_args
import types
import time
import traceback

from coupledbubble_control.rl import PPO
from coupledbubble_control.envs import PosNBC1D
from experiment_config import *


@dataclass
class Hyperparameters:
    """Hyperparameters of PPO algorithm"""
    num_envs: int = 16
    envs_per_block: int = 16
    rollout_steps: int = 16
    num_update_epochs: int = 16
    mini_batch_size: int = 512
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 2.5
    target_kl: Optional[float] = None
    norm_adv: bool = False
    policy: Literal["Beta", "Gaussian"] = "Beta"


@dataclass
class NetArch:
    """ Neural Network architecture """
    hidden_dims: List[int] = field(default_factory = lambda: [256, 256])
    activations: List[Literal["relu", "tanh"]] = field(default_factory = lambda: ["tanh", "tanh"])
    shared_dims: int = 0    # Number of shared layers

    def __post_init__(self):
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        
        allowed = {"relu", "tanh"}
        if not all(a in allowed for a in self.activations):
            raise ValueError(f"activations must be in {allowed}.")
        
        if (len(self.activations)==1 and len(self.hidden_dims) > 0):
            self.activations = self.activations * len(self.hidden_dims)
        elif len(self.activations) != len(self.hidden_dims):
            raise ValueError("activations must have length 1 or match hidden_dims")
        
        if not (0 <= self.shared_dims <= len(self.hidden_dims)):
            raise ValueError("shared_dims must be between 0 and len(hidden_dims).")
        
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

@dataclass
class Config:
    runmode: Literal["train", "preview", "eval", "sample"] = "preview"
    model_name: Optional[str] = None
    exp: Experiment = field(default_factory= lambda: Experiment(
        trial_name = f"PPO_PA_PS_COUPLED_SYSTEM_CONTROL",
        total_timesteps = int(2e7)
    ))
    stat: StaticEnvFeatures = field(default_factory=StaticEnvFeatures)
    dyn: DynamicEnvFeatures = field(default_factory=DynamicEnvFeatures)
    actions: ActionSpace = field(default_factory=ActionSpace)
    observations: ObservationSpace = field(default_factory=lambda : ObservationSpace(
        target_observation_minimum = -0.5,
        target_observation_maximum =  0.5,
        bubble_observation_minimum = -0.5,
        bubble_observation_maximum =  0.5
    ))
    hpar: Hyperparameters = field(default_factory=Hyperparameters)
    net: NetArch = field(default_factory=NetArch)
    cuda_opts: CUDAOpts = field(default_factory=CUDAOpts)


def _pop_flag(argv: list[str], *names: str) -> Optional[str]:
    """
    Az első olyan flag-et (pl. --config / -c / --exp.config-file) kiveszi az argv-ből,
    amelyik szerepel benne, és visszaadja az utána következő értéket.
    """
    for flag in names:
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value")
            val = argv[i + 1]
            del argv[i : i + 2]
            return val
    return None

def _from_yaml_dataclass_old(cls, data: dict):
    """
    Rekurzívan felépít egy dataclass-t egy YAML-ból beolvasott dict-ből.
    """
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]

        # Ha a mező típusa is dataclass, és a YAML-ben dict van → építsünk belőle példányt.
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _from_yaml_dataclass(f.type, val)
        else:
            kwargs[f.name] = val

    return cls(**kwargs)


def _unwrap_optional(tp):
    origin = get_origin(tp)

    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]

    return tp


def _from_yaml_value(tp, val):
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)

    # nested dataclass
    if is_dataclass(tp) and isinstance(val, dict):
        return _from_yaml_dataclass(tp, val)

    # list[...] mezők
    if origin is list and isinstance(val, list):
        args = get_args(tp)
        item_type = args[0] if args else object
        return [_from_yaml_value(item_type, item) for item in val]

    # tuple[...] mezők, ha vannak
    if origin is tuple and isinstance(val, list):
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_yaml_value(args[0], item) for item in val)
        return tuple(
            _from_yaml_value(item_type, item)
            for item_type, item in zip(args, val)
        )

    return val


def _from_yaml_dataclass(cls, data: dict):
    type_hints = get_type_hints(cls)
    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        field_type = type_hints.get(f.name, f.type)
        kwargs[f.name] = _from_yaml_value(field_type, data[f.name])

    return cls(**kwargs)

def parse_config(argv: Optional[Sequence[str]] = None) -> Config:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    yaml_name = _pop_flag(
        argv,
        "--config",
        "-c",
        "--exp.config-file",
        "--exp.config_file",
    )

    if yaml_name is not None:
        yaml_path = Path(__file__).resolve().parent / "configs" / yaml_name
        print(yaml_path)
        #yaml_path = os.path.join(PROJECT_DIR, "configs", yaml_name)
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
        except FileNotFoundError as e:
            raise SystemExit(f"Config file not found: {e}")

        base_config = _from_yaml_dataclass(Config, yaml_data)

        if hasattr(base_config, "exp") and hasattr(base_config.exp, "config_file"):
            base_config.exp.config_file = yaml_name

        return tyro.cli(Config, default=base_config, args=argv)
    
    return tyro.cli(Config, args=argv)


def make_envs(config, 
              collect_trajectories: bool = False,
              trajectory_dir: str = "./",
              device: str = "cuda") -> PosNBC1D:
    
    num_bubbles = config.stat.number_of_bubbles
    num_harmonics = config.dyn.number_of_harmonics
    trajectory_resolution = config.exp.trajectory_resolution if collect_trajectories else 0

    return PosNBC1D(
        num_bubbles=num_bubbles,
        num_envs=config.hpar.num_envs,
        envs_per_block=config.hpar.envs_per_block,
        # Action & Observation Spaces
        action_space_schema=config.actions.get_space(num_harmonics),
        observation_space_schema=config.observations.get_space(num_bubbles),
        # Physical Parameters
        R0=config.stat.equilibrium_radius,
        PA=config.dyn.pressure_amps,
        FR=config.dyn.excitation_freqs,
        PS=config.dyn.phase_shift,
        # Acoustic Field
        num_components=num_harmonics,
        acoustic_field=config.dyn.acoustic_field_type,
        # Static Features / Scene Setup
        episode_length=config.stat.max_steps_per_episode,
        time_step_length=config.stat.time_step_length,
        initial_position=config.stat.initial_position,
        initial_distance=config.stat.initial_distance,
        target_position=config.stat.target_position,
        target_distance=config.stat.final_distance,
        alignment=config.stat.alignment,
        min_distance=min(config.stat.distance_range),
        max_distance=max(config.stat.distance_range),
        # Static Features / Reward Properties
        dnorm=config.stat.dnorm,
        apply_termination=config.stat.apply_termination,
        position_tolerance=config.stat.position_tolerance,
        positive_terminal_reward=config.stat.positive_terminal_reward,
        negative_terminal_reward=config.stat.negative_terminal_reward,
        rewards_weights=config.stat.reward_weights,
        reward_exp=config.stat.reward_shape_exp,
        # Experiement Properties
        seed=config.exp.seed,
        collect_trajectories=collect_trajectories,
        trajectory_resolution=trajectory_resolution,
        trajectory_buffer_store=str(trajectory_dir),
        render_env=config.exp.render_env,
        # CUDA-backed-ops
        backend=config.cuda_opts.backend,
        variant=config.cuda_opts.variant,
        max_kernel_steps=config.cuda_opts.max_kernel_steps,
        kernel_steps=config.cuda_opts.kernel_steps,
        compiler=config.cuda_opts.cupy_compiler,
        max_registers=config.cuda_opts.max_registers,
        cuda_mode="release",
        device=device
    )


def make_model(config, venvs, model_dir):
    model = PPO(
        venvs=venvs,
        learning_rate=config.hpar.learning_rate,
        gamma=config.hpar.gamma,
        gae_lambda=config.hpar.gae_lambda,
        mini_batch_size=config.hpar.mini_batch_size,
        clip_coef=config.hpar.clip_coef,
        clip_vloss=config.hpar.clip_vloss,
        ent_coef=config.hpar.ent_coef,
        vf_coef=config.hpar.vf_coef,
        max_grad_norm=config.hpar.max_grad_norm,
        target_kl=config.hpar.target_kl,
        norm_adv=config.hpar.norm_adv,
        rollout_steps=config.hpar.rollout_steps,
        num_update_epochs=config.hpar.num_update_epochs,
        cuda=True,
        torch_deterministic=True,
        seed=config.exp.seed,
        net_archs=config.net.to_dict(),
        policy_type=config.hpar.policy
    )

    if config.model_name is not None:
        model.load_model(config.model_name, model_dir)

    return model

def preview(config):
    print("Environment simulation preview")
    config.exp.render_env = True
    venvs = make_envs(config, False, "Preview")
    venvs.reset()

    try:
        for _ in range(20000):
            obs, rew, term, trunc, info = venvs.step()
            #input()
    except KeyboardInterrupt:
        print("Simulation terminated by the user")


def train(config):
    trial_name = config.exp.trial_name + \
        f"_b{config.stat.number_of_bubbles:d}_epb{config.hpar.envs_per_block}_ne{config.hpar.num_envs}_id{int(time.time())}"
    root_dir  = Path(__file__).resolve().parent
    model_dir = root_dir / "models"
    metrics_dir = root_dir / "metrics"

    train_venvs = make_envs(
        config,
        collect_trajectories=False,
        device="cuda"
    )

    model = make_model(config, train_venvs, model_dir)

    learn_kwargs = {}
    config.exp.log_training = True
    if config.exp.log_training:
        run_dir = root_dir / "runs" / config.exp.project_name
        run_dir.mkdir(exist_ok=True, parents=True)

        learn_kwargs.update(
            dict(
                log_dir=str(run_dir),
                project_name=config.exp.project_name,
                trial_name=trial_name,
                log_frequency=config.exp.log_frequency
            )
        )

    save_model = False
    try:
        model.learn(config.exp.total_timesteps, **learn_kwargs)
        save_model = True

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        try:
            answer = input("Do you want to save the model? [y/N]: ").strip().lower()
            save_model = answer in ("y", "yes")
        except EOFError:
            # pl. ha nincs stdin (pl. notebook / pipe)
            print("No input available, skipping save.")
            save_model = False
    except Exception:
        print("Error during training:")
        traceback.print_exc()
        raise
    finally:
        if save_model:
            model.save_model(trial_name, model_dir)

    if save_model:
        # Collect Statistics for saved model
        try:
            model.predict(
                total_episodes=config.exp.eval_episodes,
                metrics_dir=metrics_dir,
                metrics_fname=trial_name)
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")


def eval(config):
    if config.model_name is not None:
        trial_name = config.model_name
    else:
        trial_name = config.exp.trial_name + \
            f"_b{config.stat.number_of_bubbles:d}_epb{config.hpar.envs_per_block}_ne{config.hpar.num_envs}_id{int(time.time())}"
    root_dir  = Path(__file__).resolve().parent
    model_dir = root_dir / "models"
    metrics_dir = root_dir / "metrics"

    eval_venvs = make_envs(
        config,
        collect_trajectories=False,
        device="cuda"
    )

    model = make_model(config, eval_venvs, model_dir)

    try:
        model.predict(
            total_episodes=config.exp.eval_episodes,
            metrics_dir=metrics_dir,
            metrics_fname=trial_name)
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")


def sample_trj(config):
    if config.model_name is not None:
        trial_name = config.model_name
    else:
        trial_name = config.exp.trial_name + \
            f"_b{config.stat.number_of_bubbles:d}_epb{config.hpar.envs_per_block}_ne{config.hpar.num_envs}_id{int(time.time())}"
    root_dir  = Path(__file__).resolve().parent
    model_dir = root_dir / "models"
    trj_dir   = root_dir.parents[2] / "trj" / config.exp.project_name / trial_name

    print(trj_dir)
    input()

    eval_venvs = make_envs(
        config,
        collect_trajectories=True,
        trajectory_dir=trj_dir,
        device="cuda"
    )

    model = make_model(config, eval_venvs, model_dir)

    try:
        model.predict(
            total_episodes=config.exp.num_saved_trajectories)
    except KeyboardInterrupt:
        print("Trajectory sampling interrupted by user.") 

if __name__ == "__main__":
    config = parse_config()
    print(config)

    if config.runmode == "train":
        # Train and evaluate
        train(config)

    elif config.runmode == "eval":
        # Evaluate training performance
        eval(config)

    elif config.runmode == "sample":
        # Sample Random trajectories
        sample_trj(config)

    elif config.runmode == "preview":
        preview(config)
    else:
        print("Unknown runmode")


