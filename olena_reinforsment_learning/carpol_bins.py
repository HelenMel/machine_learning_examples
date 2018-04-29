import gym
import numpy as np
import matplotlib.pyplot as plt

class StateTransformer:
    def __init__(self):
        self.cart_position_range = np.linspace(-1.25, 1.25, 9)
        self.cart_velocity_range = np.linspace(-2, 2, 9)
        self.pole_angle_range = np.linspace(-0.25, 0.25, 9)
        self.pole_velocity_range = np.linspace(-3.5, 3.5, 9)

    def merge_ids(self, ids):
        return int("".join([str(int(f)) for f in ids]))

    def value_to_bin_id(self, value, values_range):
        return np.digitize(x=[value], bins=values_range)[0]

    def transform(self, state):
        pos, vel, angle, pole_vel = state
        return self.merge_ids([
            self.value_to_bin_id(pos, self.cart_position_range),
            self.value_to_bin_id(vel, self.cart_velocity_range),
            self.value_to_bin_id(angle, self.pole_angle_range),
            self.value_to_bin_id(pole_vel, self.pole_velocity_range)
        ])

class QModel:
    #   helps update Q table
    #   helps choose next state and randomize choice
    def __init__(self, state_transformer, env):
        # uniformly distributed matrix
        # there are 2 actions and 4 values in state(10 bins each)
        n_states = 10 ** env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))
        self.state_transformer = state_transformer
        self.env = env


    def predict(self, state):
        unique_id = self.state_transformer.transform(state)
        return self.Q[unique_id]

    def update(self, state, action, G):
        unique_id = self.state_transformer.transform(state)
        alpha = 0.01
        self.Q[unique_id, action] += alpha * (G - self.Q[unique_id, action])

    # help to return action and randomize it a bit
    def sample_action(self, state, eps):
        if np.random.random() < eps:
            #return random action
            return self.env.action_space.sample()
        else:
            all_actions_from_state = self.predict(state)
            return np.argmax(all_actions_from_state)

def play_one(env, model, eps, gamma):
    state = env.reset()

    done = None
    t = 0
    total_reward = 0
    while not done and t < 10000:
        action = model.sample_action(state, eps)
        prev_state = state
        state, reward, done, info = env.step(action)

        total_reward += reward

        # we have a penalty if state finished too fast
        if done and t < 199:
            reward = -300

        next_Q = np.max(model.predict(state))
        G = reward + gamma * next_Q
        model.update(prev_state, action, G)

        t+= 1
    return total_reward

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running average")
    plt.show()

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_transformer = StateTransformer()
    model = QModel(state_transformer, env)
    gamma = 0.9

    N = 10000
    total_rewards = np.empty(N)
    for n in range(N):
        # eps become smaller each iteration because we learned space
        eps = 1.0 / np.sqrt(n + 1)
        r = play_one(env, model, eps, gamma)
        total_rewards[n] = r
        if n % 100 == 0:
            print("Episode: ", n, " reward: ", r, " eps: ", eps)
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

