import gym
import numpy as np


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    best_weights = np.random.rand(4)
    best_time = 0
    for trial in range(100):
        new_weights = np.random.rand(4)
        avg_t = []
        for episodes in range(1000):
            state = env.reset()
            t = 0; done = False
            while not done:
                predict = new_weights.dot(state)
                action = 1 if predict > 0 else 0
                state, reward, done,_ = env.step(action)
                t += 1
            avg_t.append(t)
        if np.sum(avg_t) / 1000.0 > best_time:
            best_weights = new_weights
            best_time = np.sum(avg_t) / 1000.0
    # what is the weights? how to use it in game?

    # test
    new_weights = best_weights
    state = env.reset()
    t = 0; done = False
    while not done:
        predict = new_weights.dot(state)
        action = 1 if predict > 0.5 else 0
        state, reward, done, _ = env.step(action)
        t += 1
    print("best: {0}", t)