## In this file I'm try to implement Q-learning using
## stohastic gradient descent and RBFSampler as a kernels
## we don't use Q table anymore but it is still Q learning
## because we learn online, each step use a prediction for a next step
## and because we have an exploration phase to encourage learning

import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

from carpol_bins import plot_running_avg
import matplotlib.pyplot as plt

# sklearn has its own SGDRegressor but here is custom implementation for a learning purpose
class MySGDRegressor:
    def __init__(self, dimensions, learning_rate):
        self.weights = np.random.random(dimensions) / np.sqrt(dimensions)
        self.lr = learning_rate

    def fit(self, X, Y):
        pass

    def partial_fit(self, X, Y):
        observed = Y
        predicted = X.dot(self.weights)
        self.weights += self.lr * (observed - predicted).dot(X)

    def predict(self, X):
        return X.dot(self.weights)

class StateTransformer:
    def __init__(self):
        # this doesn't make sense to get a samples for a kernels from enviroment random variables
        # because they return some states that are almost impossible to get
        N_examples = 20000
        N_state_dimensions = 4
        # range of examples -2 to 2
        examples = np.random.random((N_examples, N_state_dimensions)) * 2 - 2
        scaler = StandardScaler()
        scaler.fit(examples)

        _kernels = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000)),
        ])
        kernel_samples = _kernels.fit_transform(scaler.transform(examples))

        # we need number of dimensions to create MySGDRegressor
        self.dimensions = kernel_samples.shape[1]
        self.scaler = scaler
        self.kernels = _kernels

    def transform(self, state):
        scaled = self.scaler.transform(state)
        return self.kernels.transform(scaled)

class QLearningModel:
    ## initialize variables to store multiple models - one model per each action
    ## this models helps to approximate Q table and predict next action
    def __init__(self, env, state_transformer, learning_rate=0.01):
        self.state_transformer = state_transformer
        action_models = []
        for a in range(env.action_space.n):
            model = MySGDRegressor(state_transformer.dimensions, learning_rate)
            action_models.append(model)
        self.action_models = action_models
        self.env = env

    def predict(self, state):
        X = self.state_transformer.transform(np.atleast_2d(state))
        return np.stack([m.predict(X) for m in self.action_models]).T

    def update(self, state, action, G):
        X = self.state_transformer.transform(np.atleast_2d(state))
        self.action_models[action].partial_fit(X, [G])

    def next_action(self, state, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

def play_one(env, model, eps, gamma):
    state = env.reset()
    total_rewards = 0
    done = False
    iters = 0
    while not done and iters < 2000:
        action = model.next_action(state, eps)
        prev_state = state
        state, reward, done, info = env.step(action)

        if done:
            # This is very important to give a negative reward after episode done
            # So episodes that last less get the smallest reward
            reward = -200
        # only reward that is not done
        if reward == 1:
            total_rewards += reward
        #update model
        next = model.predict(state)
        assert (next.shape == (1, env.action_space.n))
        G = reward + gamma * np.max(next)
        model.update(prev_state, action, G)

        iters += 1
    return total_rewards

def main():
    env = gym.make('CartPole-v0')
    st = StateTransformer()
    model = QLearningModel(env, st)
    gamma = 0.99

    N_episodes = 500
    total_rewards = np.empty(N_episodes)
    for n in range(N_episodes):
        # decay epsilon to decrease exploration
        eps = 1.0 / np.sqrt(n + 1)
        reward = play_one(env, model, eps, gamma)
        total_rewards[n] = reward
        if n % 100 == 0:
            print("episode: ", n, "reward:", reward)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

if  __name__ == '__main__':
    main()