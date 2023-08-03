import numpy as np
from PyFlyt.core import Aviary

start_pos = np.array([[0.0, 0.0, 5.0]])
start_orn = np.zeros_like(start_pos)


drone_options = dict()
drone_options["model_dir"] = "pyflyt_dogfight/models"
drone_options["drone_model"] = "aggressor"

# start the environment
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type="fixedwing",
    drone_options=drone_options,
)

step = 0
while True:
    env.step()

    if step < 100:
        env.set_setpoint(0, np.array([0.0, 0.0, 0.0, 1.0]))
    else:
        env.set_setpoint(0, np.array([0.0, -1.0, 0.0, 1.0]))

    step += 1
