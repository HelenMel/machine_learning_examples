import gym
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

import matplotlib
import matplotlib.pyplot as plt

# https://github.com/openai/gym/wiki/MountainCar-v0
# Mountain car hav only 2 states - position and velocity
# And 3 actions -push left, push right and no push
# I'll use RBFSampler to partially fit action prediction function
# each action should be a separate kernel
# In my files I called it states instead of features - it is easier for me to be closer to
# reinforcement learning algorithm

# this transformer helps us to limit range of possible states
# also RBF samples helps to generate initial kernel values
# after that all other states will show this distance from initial kernel
# different kernels should represent different features of state space
# in this case we have more features, than a states
class StateTransformer:
    def __init__(self, env, n_components=500):
        # position has a scale -1.2 to 0.6
        # velocity has a scale -0.07 to 0.07
        # we have to standardize them
        # transform it to np.array because this is type of standart scaler input
        random_state_samples = np.array([env.observation_space.sample() for x in range(10000)])
        _scaler = StandardScaler()
        _scaler.fit(random_state_samples)

        # n_components is the number of exemplars
        _kernels = FeatureUnion([
            ("rbf1", RBFSampler(5.0, n_components=n_components)),
            ("rbf2", RBFSampler(2.0, n_components=n_components)),
            ("rbf3", RBFSampler(1.0, n_components=n_components)),
            ("rbf4", RBFSampler(0.5, n_components=n_components))
        ])
        # init kernels with random samples
        examples = _kernels.fit_transform(_scaler.transform(random_state_samples))
        self.dimensions = examples.shape[1]
        self.scaler = _scaler
        self.kernels = _kernels

    def transform(self, state):
        # to return feature we have to
        # 1) transform state values to standard scale
        scaled = self.scaler.transform(state)
        # 2) create a features according to a distance from center of each kernel
        return self.kernels.transform(scaled)

# this class helps us predict next action and update Q given current experience
# main difference with a bin model is that
# for each action we use a separate model
class RBFLearnModel:
    def __init__(self, env, state_transformer, learning_rate):
        self.env = env
        self.state_transformer = state_transformer
        self.action_models = []
        for action_n in range(env.action_space.n):
            # this is the same stochastic gradient descent that we used in a previous model
            model = SGDRegressor(learning_rate=learning_rate)
            # we have to init each action model before use
            # in the other case result of the fist prediction will be terribly wrong
            random_state = state_transformer.transform([env.reset()])
            X =  random_state
            model.partial_fit(X, [0])
            self.action_models.append(model)

    # returns all possible actions from Q table for this state
    # use np.array to make easier to find argmax
    def predict(self, state):
        X = self.state_transformer.transform([state])
        # for each action return possible outcome
        return np.array([m.predict(X)[0] for m in self.action_models])

    # action models should be updated using partial fit
    def update(self, state, action, G):
        X = self.state_transformer.transform([state])
        # sklearn expect y to be a one dimentional vector, but it is a scalar
        self.action_models[action].partial_fit(X, [G])

    # next action should be partially random action
    # if it is not random it is maximised for a current state
    def next_action(self, state, eps = 0.01):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

# this is one episode of emulator
def play_one(env, model, eps, gamma):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = model.next_action(state, eps)
        old_state = state
        state, reward, done, info = env.step(action)

        # after step is done, we have to improve our model
        G = reward + gamma * np.max(model.predict(state)[0])
        model.update(old_state, action, G)

        total_reward += reward

    return total_reward

# it shows for a different positions and velocity confidence in prediction
def plot_cost_to_go(env, estimator, num_tiles=20):
    # position axis
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_tiles)
    # velocity axis
    y  = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda  _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == - V(s)')
    ax.set_title("Cost-ToGO Function")

    fig.colorbar(surf)
    plt.show()

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t - 100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = StateTransformer(env)
    model = RBFLearnModel(env, ft, "constant")
    gamma = 0.99

    N_episodes = 300
    total_rewards = np.empty(N_episodes)
    for episode in range(N_episodes):
        eps = 0.1 * (0.97 ** episode)
        one_episode_reward = play_one(env, model, eps, gamma)
        total_rewards[episode] = one_episode_reward
        if (episode + 1) % 100 == 0:
            print("episode:", episode, "total reward:", total_rewards)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
    plot_cost_to_go(env, model)
