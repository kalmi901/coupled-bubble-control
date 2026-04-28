from __future__ import annotations
import torch
from typing import Dict

def process_final_observation(next_obs: torch.Tensor, infos: Dict):
    real_next_obs = next_obs.clone()
    if 'final_observation' in infos.keys():
        real_next_obs[infos['dones']] = infos['final_observation']

    return real_next_obs