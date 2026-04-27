from __future__ import annotations
import dev_bootstap
from coupledbubble_control.envs import PosNBC1D
from coupledbubble_control.backends import BackendName, KernelVariant
from pathlib import Path


if __name__ == "__main__":
    print("test-vec-env")

    root = Path(__file__).resolve().parent
    buffer_store = root / "test_buffer_store"

    print(buffer_store)

    env = PosNBC1D(
        num_bubbles = 4,
        num_envs = 1024,
        envs_per_block = 32,
        #initial_position= [-0.01, 0.01],
        initial_position = "random",
        target_position = [-0.25, 0.0, 0.1, 0.25],
        render_env = False,
        acoustic_field = "SW_N",
        episode_length = 128,
        backend=BackendName.NUMBA,
        variant=KernelVariant.WARP,
        collect_trajectories=True,
        trajectory_buffer_store=str(buffer_store)
    )

    #print(env.action_space.sample())
    #print(env.observation_space.sample())

    env.reset()
    input()

    for steps in range(100):
        env.step(action=None)
        #input()