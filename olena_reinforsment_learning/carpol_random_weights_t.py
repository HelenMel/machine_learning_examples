import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

def get_action(state, weights):
    # in my version I tries to compare with 0.5
    return 1 if state.dot(weights) > 0 else 0

def play_episode(env, weights):
    state = env.reset()
    done = False
    t = 0
    while not done and t < 10000:
        t += 1
        action = get_action(state, weights)
        state, reward, done, info = env.step(action)
        if done:
            break
    return t

def play_one_trial(env, trial_length, weights):
    # I called it average time
    episode_lenghts = np.empty(trial_length)
    for trial in range(trial_length):
        episode_lenghts[trial] = play_episode(env, weights)
    avg_episode_length = episode_lenghts.mean()
    print("avg length:", avg_episode_length)
    return avg_episode_length

def random_weights(env):
    all_trials_length = []
    best = 0
    best_weights = None
    for i in range(100):
        new_weights = np.random.random(4) * 2 - 1
        avg_episode_length = play_one_trial(env, 100, new_weights)
        all_trials_length.append(avg_episode_length)

        if avg_episode_length > best:
            best = avg_episode_length
            best_weights = new_weights
    return all_trials_length, best_weights

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, "test_result", force=True)
    all_trials_length, best_weights = random_weights(env)
    plt.plot(all_trials_length)
    plt.show()

    print("Test game")
    test_time = play_one_trial(env, 100, best_weights)
    print("time", test_time)
