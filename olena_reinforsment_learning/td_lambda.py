import gym
import numpy as np
import matplotlib.pyplot as plt


import mountain_car_q_learning
from mountain_car_q_learning import plot_cost_to_go, StateTransformer, RBFLearnModel, plot_running_avg

class BaseModel:
    def __init__(self, dimensions):
        self.weights = np.random.randn(dimensions) / np.sqrt(dimensions)

    def partial_fit(self, X, Y, eligibility, learning_rate=0.01):
        self.weights += learning_rate * (Y - X.dot(self.weights)) * eligibility

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.weights)

class TDLambdaModel:
    def __init__(self, env, state_transformer):
        self.env = env
        self.models = []
        self.state_transformer = state_transformer

        dimensions = state_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, dimensions))
        for a in range(env.action_space.n):
            model = BaseModel(dimensions)
            self.models.append(model)

    def predict(self, state):
        X = self.state_transformer.transform([state])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X) for m in self.models])

    def update(self, state, action, G, gamma, lambda_):
        model = self.models[action]
        X = self.state_transformer.transform([state])
        # have to update eligibilities based on previos experience
        self.eligibilities *= gamma * lambda_
        # gradient is feature vector itself (X[0])
        self.eligibilities[action] += X[0]

        model.partial_fit(X[0], [G], self.eligibilities[action])

    def next_action(self, state, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

def play_one(env, model, eps, gamma, lambda_):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.next_action(state, eps)
        prev_state = state
        state, reward, done, info = env.step(action)

        G = reward + gamma*np.max(model.predict(state)[0])
        model.update(prev_state, action, G, gamma, lambda_)

        total_reward += reward

    return total_reward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state_transformer = StateTransformer(env)
    model = TDLambdaModel(env, state_transformer)
    gamma = 0.99
    lambda_ = 0.7

    N = 300
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97 ** n)
        total_reward = play_one(env, model, eps, gamma, lambda_)
        total_rewards[n] = total_reward
        if n % 50 == 0:
            print("episode:", n, "total reward:", total_reward)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
    plot_cost_to_go(env, model)

