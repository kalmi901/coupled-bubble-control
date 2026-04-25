from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


# ---- Helper Functions ----
ACTIVATIONS = {
    "relu" : nn.ReLU(),
    "tanh" : nn.Tanh(),
    "leaky_relu" : nn.LeakyReLU(),
    None: nn.Identity(),
}

def get_activation(name: str) -> nn.Module:
    try: 
        return ACTIVATIONS[name.lower()] if isinstance(name, str) else ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"Invalid activation: {name}")

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def build_network(dims, activations):
    layers = []
    for i, act in enumerate(activations):
        layer = init_layer(nn.Linear(dims[i], dims[i+1]))
        layers.append(layer)
        layers.append(get_activation(act))
    
    return nn.Sequential(*layers)


class ActorCriticGaussianPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str], 
                 hidden_dims: List[int],
                 action_high: torch.Tensor,
                 action_low : torch.Tensor,
                 shared_dims: int = 0,
                 log_std_max: float = -1.0,
                 log_std_min: float = -10.0,
                 log_std_init: float = -2.0,
                 **kwargs) -> None:
        super().__init__()

        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._log_std = nn.Parameter(torch.ones(output_dim) * log_std_init, requires_grad=True)

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias",  (action_high + action_low) / 2.0)

        dims = [input_dim] + hidden_dims + [output_dim]
        acts = activations + ["tanh"]     # Last Activation

        if shared_dims == 0:
                self.features = nn.Identity()
        else:
            s_dims = dims[0:shared_dims+1]
            s_acts = acts[0:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.pi = build_network(dims[shared_dims:], acts[shared_dims:])                     # Policy Network
        self.vf = build_network(dims[shared_dims:-1]+[1], acts[shared_dims:-1]+[None])      # Value Network

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.vf(self.features(x))
    
    def action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean = self.pi(self.features(x))
        mean = mean * self.action_scale + self.action_bias 

        if deterministic:
            return mean
        
        log_std = self._log_std.clamp(self._log_std_min, self._log_std_max).expand_as(mean)
        std = log_std.exp() * self.action_scale

        return torch.distributions.Normal(mean, std).rsample()
    
    def forward(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        f = self.features(x)
        mean = self.pi(f)
        mean = mean * self.action_scale + self.action_bias
        log_std = self._log_std.clamp(self._log_std_min, self._log_std_max).expand_as(mean)
        std = log_std.exp() * self.action_scale

        probs = torch.distributions.Normal(mean, std)
        if a is None:
            a = probs.rsample()

        return a, probs.log_prob(a).sum(1), probs.entropy().sum(1), self.vf(f)
    

class ActorCriticBetaPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations : List[str],
                 hidden_dims: List[int],
                 action_high: torch.Tensor,
                 action_low: torch.Tensor,
                 shared_dims: int = 0,
                 **kwargs) -> None:
        super().__init__()

        self.register_buffer("action_scale", (action_high - action_low))
        self.register_buffer("action_bias",     action_low)

        dims = [input_dim] + hidden_dims + [2 * output_dim]
        acts = activations + [None]

        if shared_dims == 0:
            self.features = nn.Identity()

        else:
            s_dims = dims[:shared_dims+1]
            s_acts = acts[:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.pi = build_network(dims[shared_dims:], acts[shared_dims:])        # Policy Network
        self.vf = build_network(dims[shared_dims:], acts[shared_dims:])        # Value Network

    
    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.vf(self.features(x))
    

    def forward(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        f = self.features(x)
        params = self.pi(f)

        alpha, beta = torch.chunk(params, 2, dim=-1)

        alpha = torch.nn.functional.softplus(alpha) + 1.0  # Avoid instabilities
        beta = torch.nn.functional.softplus(beta)   + 1.0

        # Beta distribution
        probs = torch.distributions.Beta(alpha, beta)

        if a is None:
            a_unscaled = probs.rsample()

        else:
            a_unscaled = (a - self.action_bias) / self.action_scale

        return a_unscaled * self.action_scale + self.action_bias, probs.log_prob(a_unscaled).sum(1), probs.entropy().sum(1), self.vf(f)



if __name__ == "__main__":
    input_dim = 4
    output_dim = 2
    activations = ["relu", "relu", "relu"]

    hidden_dims = [120, 120, 84]
    high = torch.full((4,), 2)
    low = torch.full((4,), 0)

    nn1 = ActorCriticBetaPolicy(input_dim, output_dim, activations, hidden_dims,
                                high, low, 2)
    

    print(nn1)

    nn2 = ActorCriticGaussianPolicy(input_dim, output_dim, activations, hidden_dims,
                                high, low, 1)
    
    print(nn2)