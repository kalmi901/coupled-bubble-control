from __future__ import annotations
import re
from pathlib import Path
import os
import zarr

class TrajectoryZarrReader:
    SHARD_RE = re.compile(r"^shard_(\d{5})_(\d{5})\.zarr$")
    TRJ_RE = re.compile(r"^trj_(\d{6})$")

    def __init__(
            self,
            root_dir: str) -> None:
        
        self.root_dir = root_dir
        #self.global_metadata = self._load_global_metadata()
        self.index = self._build_index()
        self.episode_ids = sorted(self.index.keys())


    def _load_global_metadata(self):
        meta_path = os.path.join(self.root_dir, "global_metadata.zarr")
        if not os.path.exists(meta_path):
            return {}

        root = zarr.open(meta_path, mode="r")
        return dict(root.attrs)

    def _build_index(self):
        index = {}
        for name in os.listdir(self.root_dir):
            m = self.SHARD_RE.match(name)
            if not m:
                continue
            shard_path = os.path.join(self.root_dir, name)
            root = zarr.open(shard_path, mode='r')

            for group_name in root.group_keys():    # type: ignore
                m2 = self.TRJ_RE.match(group_name)
                if not m2:
                    continue
                episode_id = int(m2.group(1))

                index[episode_id] = (shard_path, group_name)
        return index
    
    def _get_group(self, episode_id: int):
        if episode_id not in self.index:
            raise KeyError(f"Episode {episode_id} not found")
        shard_path, group_name = self.index[episode_id]
        root = zarr.open(shard_path, mode="r")
        return root[group_name]

    def __len__(self):
        return len(self.episode_ids)
    
    def available_episodes(self):
        return list(self.episode_ids)
    
    def load_episode(self, episode_id: int):
        g = self._get_group(episode_id)
        data = {
            "observations"  : g["observations"][:],         # type: ignore
            "actions"       : g["actions"][:],              # type: ignore
            "rewards"       : g["rewards"][:],              # type: ignore
            "dense_time"    : g["dense_time"][:],           # type: ignore
            "equilibrium_radii" : g["equilibrium_radii"],   # type: ignore
            "dense_state"   : {},
            "attrs"         : dict(g.attrs)                 # type: ignore
        }

        for name, arr in g.arrays():    # type: ignore
            if name.startswith("dense_state_"):
                idx = int(name.split("_")[-1])
                data["dense_state"][idx] = arr[:]

        return data