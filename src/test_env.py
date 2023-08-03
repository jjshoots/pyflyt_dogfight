import numpy as np

from pyflyt_dogfight import DogfightEnv

env = DogfightEnv(render=True)

obs, info = env.reset()

print(obs.shape)

term, trunc = False, False

while not (term or trunc):
    actions = np.stack(
        [env.action_space.sample() for _ in range(env.num_drones)], axis=0
    )
    obs, rew, term, trunc, info = env.step(actions)
    print(rew)
    term = term.any()
    trunc = trunc.any()


print(info)
