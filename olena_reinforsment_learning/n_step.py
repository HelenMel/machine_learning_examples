import gym
import numpy as np
import matplotlib.pyplot as plt

import mountain_car_q_learning
from mountain_car_q_learning import plot_cost_to_go, StateTransformer, RBFLearnModel, plot_running_avg


# the main difference with a basic Q-learning that we calculate G for a multiple steps
class SGDRegressor:
    def __init__(self, **kwargs):
        self.weights = None
        self.learning_rate = 0.01

    # keep in mind that Y is G
    def partial_fit(self, X, Y):
        if self.weights is None:
            dimensions = X.shape[1]
            self.weights = np.random.randn(dimensions) / np.sqrt(dimensions)
        self.weights += self.learning_rate * (Y - X.dot(self.weights)).dot(X)

    def predict(self, X):
        return X.dot(self.weights)

mountain_car_q_learning.SGDRegressor = SGDRegressor

# we have to keep N states, rewards and actions
def play_one(env, model, eps, gamma, n=5):
    state = env.reset()
    done = False
    total_reward = 0
    rewards = []
    states = []
    actions = []
    # for each step we have to multiply by additional gamma
    n_gammas = np.array([gamma]*n) ** np.arange(n)

    while not done:
        action = model.next_action(state, eps)

        states.append(state)
        actions.append(action)

        prev_state = state
        state, reward, done, info = env.step(action)

        rewards.append(reward)

        # update model
        if len(rewards) >= n:
            previous_returns = n_gammas.dot(rewards[-n:])
            G = previous_returns + (gamma **n)*np.max(model.predict(state)[0])
            model.update(states[-n], actions[-n], G)

        total_reward += reward

    rewards = rewards[-n+1:]
    states = states[-n+1:]
    actions = actions[-n+1:]

    # according to documentation goal achived if position > 0.5
    win = state[0] >= 0.5
    if win:
        while len(rewards) > 0:
            G = n_gammas[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            states.pop(0)
            actions.pop(0)
            rewards.pop(0)
    else:
        # we lose, so it is a good idea to set negative reward
        while len(rewards) > 0:
            guess_rewards = rewards + [-1]*(n - len(rewards))
            G = n_gammas.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            states.pop(0)
            actions.pop(0)
            rewards.pop(0)
    return total_reward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state_transformer = StateTransformer(env)
    model = RBFLearnModel(env, state_transformer, "constant")
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



