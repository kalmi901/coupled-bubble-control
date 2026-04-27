from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, DefaultDict, Any, Optional, Sequence, Dict
import numpy as np
from numpy.typing import NDArray
import zarr
from pathlib import Path
import time

import torch

import matplotlib.pyplot as plt

@dataclass
class Trajectory:
    env_id: int
    num_units: int
    equilibrium_radii: Optional[NDArray] = None

    observations: List[NDArray] = field(default_factory=list)
    actions: List[NDArray] = field(default_factory=list)
    rewards: List[NDArray] = field(default_factory=list)
    dense_time: List[NDArray] = field(default_factory=list)
    dense_states: DefaultDict[int | str, List[NDArray]] = field(default_factory=lambda: defaultdict(list))

    episode_length: int = 0
    episode_reward: float = 0
    time_step_length: float = 5.0           # Timespan of one RL-step

    def __post_init__(self):
        if self.equilibrium_radii is None:
            # 0 --> not recorded
            self.equilibrium_radii = np.zeros(self.num_units, dtype=np.float32)


    def merge_list(self) -> Dict[Any, Any]:
        obs = np.vstack(self.observations)     # (T+1, obs_dim)
        act = np.vstack(self.actions)          # (T, acts_dim)
        rew = np.vstack(self.rewards)          # (T, 1)

        time = np.concatenate([time[1:] + i * self.time_step_length if i !=0 else time for i, time in enumerate(self.dense_time)])
    
        states = {}
        for key, dense_state in self.dense_states.items():
            states[key] = np.concatenate([state[1:,:] if i !=0 else state for i, state in enumerate(dense_state)])

        return dict(
            observations=obs,
            actions=act,
            rewards=rew,
            dense_time=time,
            dense_state=states
        )


    def clear(self) -> None:
        if self.equilibrium_radii is not None:
            self.equilibrium_radii.fill(0.0)
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dense_time.clear()
        self.dense_states.clear()
        self.episode_length = 0
        self.episode_reward = 0.0



class TrajectorBuffer:
    def __init__(
            self,
            num_envs: int,
            num_units: int,
            saved_states: Optional[Sequence[int]] = None,
            *,
            zarr_root: str,
            zarr_shard_size: int = 1000,
            metadata: Optional[Dict] = None) -> None:
        
        self.episode_count = 0
        self._num_envs = num_envs
        self._num_units = num_units
        self._state_index: List[int] = list(saved_states) if saved_states is not None else [0, 1]

        self._trajector_container: Dict[int, Trajectory] = {
            env_id: Trajectory(env_id=env_id, num_units=self._num_units)
            for env_id in range(self._num_envs)
        }

        # Create ZarrStorage Writer
        self._zarr_writer = TrajectoryZarrWriter(zarr_root, zarr_shard_size, metadata)

    
    def step(
            self,
            observations: torch.Tensor,     # shape (num_envs, ...)
            actions: torch.Tensor,          # shape (num_envs, ...)
            rewards: torch.Tensor,          # shape (num_envs, ) or (num_envs, 1)
            dense_index: torch.Tensor,    
            dense_time: torch.Tensor,
            dense_states: torch.Tensor
    ) -> None:
        
        obs = observations.cpu().numpy()
        act = actions.cpu().numpy()
        rew = rewards.cpu().numpy().reshape(-1)

        db_index = dense_index.cpu().numpy()
        db_time  = dense_time.cpu().numpy()
        db_states = dense_states.cpu().numpy()

        for env_id in range(self._num_envs):
            trj = self._trajector_container[env_id]
            trj.observations.append(obs[env_id])
            trj.actions.append(act[env_id])
            trj.rewards.append(rew[env_id])

            npts = db_index[env_id]
            trj.dense_time.append(db_time[:npts, env_id])
            for idx in self._state_index:
                trj.dense_states[idx].append(db_states[:npts, idx, env_id, :self._num_units])

        # ! Reset dense-buffers in-place
        dense_index.zero_()
        dense_time.zero_()
        dense_states.zero_()

    def end_episode(
            self,
            env_ids: Sequence[int],
            final_observations : torch.Tensor,
            episode_lengths,
            episode_rewards,
            equilibrium_radii):
        
        final_obs = final_observations.cpu().numpy()
        ep_len = episode_lengths.cpu().numpy()
        ep_rew = episode_rewards.cpu().numpy()
        eq_radii = np.asanyarray(equilibrium_radii, dtype=np.float32)

        print("Trj buffer end episode")

        for i, env_id in enumerate(env_ids):
            trj = self._trajector_container[int(env_id)]
            trj.observations.append(final_obs[i])
            trj.episode_length = ep_len[i]
            trj.episode_reward = ep_rew[i]
            trj.equilibrium_radii = eq_radii[i]

            episode_id = self.episode_count
            self._zarr_writer.write(episode_id, trj)
            self.episode_count +=1
            trj.clear()


class TrajectoryZarrWriter:
    def __init__(
            self,
            root_dir: str,
            shard_size: int = 1000,
            global_metadata: Optional[Dict] = None
    ) -> None:
        
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.global_metadata: Dict = global_metadata if global_metadata is not None else {}
        self._write_global_metadata()

    def _write_global_metadata(self):
        if not self.global_metadata:
            return
        meta_path = self.root_dir / "global_metadata.zarr"
        root = zarr.open(meta_path, mode="a")

        if not root.attrs.get("_initialized", False):
            root.attrs.update(self.global_metadata)
            root.attrs["_initialized"] = True  

    def _shard_path(self, episode_id: int):
        shard_start = (episode_id // self.shard_size) * self.shard_size
        shard_end = shard_start + self.shard_size - 1

        return self.root_dir / f"shard_{shard_start:05d}_{shard_end:05d}.zarr"
    
    def _create_group(self, episode_id: int):
        shard_path = self._shard_path(episode_id)
        root = zarr.open(shard_path, mode="a")
        group_name = f"trj_{episode_id:06d}"
        if group_name in root:  # type: ignore
            del root[group_name]    # type: ignore
        return root.create_group(group_name), shard_path # type: ignore

    def write(self, episode_id: int, trj: Trajectory):
        group, store_path = self._create_group(episode_id)  # type: ignore
        data = trj.merge_list()

        # MDP --
        obs = group.create_array("observations", shape=data["observations"].shape, dtype="float32")
        act = group.create_array("actions", shape=data["actions"].shape, dtype="float32")
        rew = group.create_array("rewards", shape=data["rewards"].shape, dtype="float32")
        
        obs[:] = data["observations"]
        act[:] = data["actions"]
        rew[:] = data["rewards"]

        if trj.equilibrium_radii is not None:
            eq_radii = group.create_array("equilibrium_radii", shape=trj.equilibrium_radii.shape, dtype="float32")
            eq_radii[:] = trj.equilibrium_radii

        # Dense Output --
        dense_time = group.create_array("dense_time", shape=data["dense_time"].shape, dtype="float32")
        dense_time[:] = data["dense_time"]
        for idx in data["dense_state"].keys():
            state = group.create_array(f"dense_state_{idx:0d}", shape=data["dense_state"][idx].shape, dtype="float32")
            state[:] = data["dense_state"][idx]

        # Write Meta
        group.attrs.update(dict(
            env_id=int(episode_id),
            num_units=int(trj.num_units),
            episode_length=trj.episode_length,
            episode_reward=trj.episode_reward,
            saved_unix=int(time.time())))




