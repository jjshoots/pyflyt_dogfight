import numpy as np
from PyFlyt.core import Aviary


def get_starting_pos():
    x_range = np.linspace(-5.0, 5.0, 5)
    y_range = np.linspace(-5.0, 5.0, 5)
    z_range = np.linspace(1.0, 3.0, 4)

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

    xx = xx.flatten()
    yy = yy.flatten()
    zz = zz.flatten()

    return np.stack([xx, yy, zz], axis=-1)


if __name__ == "__main__":
    # the starting position and orientations
    start_pos = get_starting_pos()
    start_orn = np.zeros_like(start_pos)

    # environment setup
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        render=True,
        drone_type="quadx",
    )

    # set to position control
    env.set_mode(7)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    while True:
        env.step()
