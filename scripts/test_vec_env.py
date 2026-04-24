from __future__ import annotations
import dev_bootstap
from coupledbubble_control.envs import PosNBC1D
from coupledbubble_control.backends import BackendName


if __name__ == "__main__":
    print("test-vec-env")

    env = PosNBC1D(
        num_bubbles = 2,
        num_envs = 8,
        envs_per_block = 2,
        initial_position= [-0.10, 0.10],
        target_position = [-0.25, 0.25],
        render_env = True,
        acoustic_field = "SW_A",
        episode_length = 25,
        backend=BackendName.NUMBA
    )

    print(env.action_space.sample())
    print(env.observation_space.sample())

    env.reset()

    for steps in range(100):
        env.step(action=None)
        #input()