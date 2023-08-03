import numpy as np

from pyflyt_dogfight import DogfightEnv

env = DogfightEnv(render=True)

obs, info = env.reset()

term, trunc = False, False

while not (term or trunc):
    actions = np.zeros((2, 4))
    actions[:, -1] = 1.0
    actions[0, 1] = -1.0
    actions[1, 1] = -1.0
    obs, rew, term, trunc, info = env.step(actions)
    term = term.any()
    trunc = trunc.any()
    print(rew)


print(info)
