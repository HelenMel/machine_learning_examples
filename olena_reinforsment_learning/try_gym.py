import gym
import numpy as np
import matplotlib.pyplot as plt

# full list on http://gym.openai.com/env
# documentation is available at
# https://github.com/openai/gym/wiki/CartPole-v0
env = gym.make('CartPole-v0')

box = env.observation_space

steps = []

positions = []
for i in range(10000):
    env.reset()

    done = False; t = 0
    while not done:
        a = env.action_space.sample()
        observation, reward, done, info = env.step(a)
        positions.append(observation[3])
        t += 1
    steps.append(t)

plt.plot(positions)
plt.show()
print(np.sum(steps) / 10000.0)
