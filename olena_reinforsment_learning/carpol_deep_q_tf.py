import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from carpol_bins import plot_running_avg

# TODO: what is the difference between this and previous version
class HiddenLayer:
    def __init__(self, n_input, n_output, activation=tf.nn.tanh, use_bias=True):
        self.Weights = tf.Variable(tf.random_normal(shape=(n_input, n_output)))
        # TODO: why to use weights
        # Tensorflow take care about variables by default in optimizer
        self.params = [self.Weights]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(n_output).astype(np.float32))
            self.params.append(self.b)
        self.activation = activation

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.Weights) + self.b
        else:
            a = tf.matmul(X, self.Weights)
        return  self.activation(a)

# TODO: we prepare Hidden layer to save and return
# a reference all the parameters, but WHY
class DeepQNet:
    # we need to create a buffer to replay an actions
    # IMPORTANT!
    # there should be possible to copy the whole deep net!
    def __init__(self,
                 dimensions,
                 n_output,
                 hidden_layer_sizes,
                 gamma,
                 max_experiences=10000, # experience replay buffer size
                 min_experiences=100, # min experiences to collect before training
                 batch_size=32): # number of samples to train

        self.dimensions = dimensions
        self.n_output = n_output
        self.layers = []
        n_input = dimensions
        for n_output_ in hidden_layer_sizes:
            layer = HiddenLayer(n_input, n_output_)
            self.layers.append(layer)
            n_input = n_output_

        # TODO: remove use_bias if something is wrong
        layer = HiddenLayer(n_input, n_output, lambda x: x)
        self.layers.append(layer)
        # layers are also a params
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # input
        # X derived from state
        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        # G is a step value - depends on a reward an previous value. Target value
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # TODO: before train we have to collect all of this in batch
        # how to save states in batch in TF?

        result = self.X
        for layer in self.layers:
            result = layer.forward(result)
        predicted_action_values = result
        self.predict_op = predicted_action_values

        # the train operation should be tricky
        selected_action_values = tf.reduce_sum(
            predicted_action_values * tf.one_hot(self.actions, n_output),
            reduction_indices=[1]
        )

        # model should predict G, so minimize diff with selected action values
        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdagradOptimizer(0.001).minimize(cost)

        self.replay_memory = {'state_before': [], 'action': [], 'reward': [], 'state_after':[]}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        ops = []
        this_params = self.params
        other_params = other.params
        for this_p, other_p in zip(this_params, other_params):
            other_value = self.session.run(other_p)
            op = this_p.assign(other_value)
            ops.append(op)
        self.session.run(ops)

    def predict(self, state):
        X = np.atleast_2d(state)
        return self.session.run(self.predict_op, feed_dict={ self.X: X })

    def train(self, target_network):
        if len(self.replay_memory['state_before']) < self.min_experiences:
            # not enough data to train
            return

        # it is important that all samples selected randomly from a memory
        idx = np.random.choice(len(self.replay_memory['state_before']),
                               size=self.batch_size,
                               replace=False)
        states_before = [self.replay_memory['state_before'][i] for i in idx]
        actions = [self.replay_memory['action'][i] for i in idx]
        rewards = [self.replay_memory['reward'][i] for i in idx]
        states_after = [self.replay_memory['state_after'][i] for i in idx]

        next_Q = np.max(target_network.predict(states_after), axis=1)
        G = [r + self.gamma*q for r, q in zip(rewards, next_Q)]

        self.session.run(
            self.train_op,
            feed_dict={
                self.X: states_before,
                self.actions: actions,
                self.G: G
            }
        )

    def update_memory(self, state_before, action, reward, state_after):
        if len(self.replay_memory['state_before']) >= self.max_experiences:
            self.replay_memory['state_before'].pop(0)
            self.replay_memory['action'].pop(0)
            self.replay_memory['state_after'].pop(0)
            self.replay_memory['reward'].pop(0)

        self.replay_memory['state_before'].append(state_before)
        self.replay_memory['state_after'].append(state_after)
        self.replay_memory['action'].append(action)
        self.replay_memory['reward'].append(reward)

    def next_action(self, state, eps):
        # greedy policy
        if np.random.random() < eps:
            return np.random.choice(self.n_output)
        else:
            X = np.atleast_2d(state)
            return np.argmax(self.predict(X)[0])



def run_once(env, model, stable_model, eps, update_period):
    # I at least have to remember what is going on
    state = env.reset()
    total_reward = 0
    done = False
    items = 0

    while not done:
        action = model.next_action(state, eps)
        state_before = state
        state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            reward = -200

        model.update_memory(state_before, action, reward, state)
        model.train(stable_model)

        items += 1
        if items % update_period == 0:
            stable_model.copy_from(model)

    return total_reward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    gamma = 0.99
    # iterations that is needed to update stable model
    update_period = 50

    dimensions = len(env.observation_space.sample())
    n_output = env.action_space.n
    hidden_layer_sizes = [200, 200]
    model = DeepQNet(dimensions, n_output, hidden_layer_sizes, gamma)
    stable_model = DeepQNet(dimensions, n_output, hidden_layer_sizes, gamma)

    # tenserflow require to initialise session before use it
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    model.set_session(session)
    stable_model.set_session(session)


    # if 'monitor' in sys.argv:
    #     filename = os.path.basename(__file__).split('.')[0]
    #     monitor_dir = './' + filename + '_' + str(datetime.now())
    #     env = wrappers.Monitor(env, monitor_dir)

    N = 500
    total_rewards = np.empty(N)

    for t in range(N):
        # epsilon should be smaller through time, to  reduce amount of explorations
        eps = 1.0 / np.sqrt(t + 1)
        reward = run_once(env, model, stable_model, eps, update_period)
        total_rewards[t] = reward
        if t % 50 == 0:
            print("episode: ", t, " reward: ", reward)

    print("total steps: ", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)










