from __future__ import annotations
import train_bootstrap
import yaml
import tyro
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import LiteralString, List, Dict, Annotated, Optional, Sequence, Literal, Any

from coupledbubble_control.rl import PPO
from coupledbubble_control.envs import PosNBC1D
from experiment_config import *


@dataclass
class Hyperparameters:
    """Hyperparameters of PPO algorithm"""
    num_envs: int = 1024
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
    runmode: Literal["train", "preview", "eval"] = "preview"
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

def _from_yaml_dataclass(cls, data: dict):
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
        print()
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



if __name__ == "__main__":
    config = parse_config()
    print(config)