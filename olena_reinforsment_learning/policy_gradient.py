import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from carpol_bins import plot_running_avg

#reuse something?
# feature transformer could be the same

# We use Neural net for policy and  linear model for value
# Warning! reward is VERY unstable here! If it found good policy at
# the beginning it will perform Well

class HiddenLayer:
    def __init__(self, n_input, n_output, activation_function=tf.nn.tanh, use_bias=True):
        self.Weights = tf.Variable(tf.random_normal(shape=(n_input, n_output)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = tf.Variable(np.zeros(n_output).astype(np.float32))
        self.activation_function = activation_function

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.Weights) + self.bias
        else:
            a = tf.matmul(X, self.Weights)
        # use activation function
        return self.activation_function(a)

# We never have this before.
class PolicyModel:
    def __init__(self, dimensions, output_size, hidden_layer_sizes):
        # create graph of actions
        n_input = dimensions
        self.layers = []
        for n_output in hidden_layer_sizes:
            layer = HiddenLayer(n_input, n_output)
            n_input = n_output
            self.layers.append(layer)

        # last layer that use softmax to make each output uses output as probablities that sum as 1
        layer = HiddenLayer(n_input, output_size, tf.nn.softmax, use_bias=False)
        self.layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        # thery are indexes to array
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        input_ = self.X
        for layer in self.layers:
            input_ = layer.forward(input_)

        # TODO is it policy a given S or probability A given S
        p_a_given_s = input_

        self.predict_op = p_a_given_s

        # TODO find out input - output shape here
        selected_probs = tf.log(
            tf.reduce_sum(
                p_a_given_s * tf.one_hot(self.actions, output_size),
                reduction_indices=[1]
            )
        )

        cost = - tf.reduce_sum(self.advantages * selected_probs)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(cost)

    # We can share the same session also for value model
    def set_session(self, session):
        self.session = session

    # X represent state
    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        feed_dict = {
            self.X: X,
            self.actions: actions,
            self.advantages: advantages
        }
        self.session.run(self.train_op, feed_dict=feed_dict)

    def predict(self, X):
        # should we reduce size of predicted item?
        X = np.atleast_2d(X)
        feed_dict = { self.X: X}
        return self.session.run(self.predict_op, feed_dict=feed_dict)

    def next_action(self, X):
        # we select action randomly, but action with higher probability has higher chances
        probabilities = self.predict(X)[0]
        return np.random.choice(len(probabilities), p=probabilities)

# Value model should use linear model
class ValueModel:
    def __init__(self, dimensions, hidden_layer_sizes):
        self.layers = []
        n_input = dimensions
        for n_output in hidden_layer_sizes:
            layer = HiddenLayer(n_input, n_output)
            self.layers.append(layer)
            n_input = n_output

        layer = HiddenLayer(n_input, 1, lambda x: x)
        self.layers.append(layer)


        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        input_ = self.X
        for layer in self.layers:
            input_ = layer.forward(input_)
        predicted_Y = tf.reshape(input_, [-1])
        self.predict_op = predicted_Y

        cost = tf.reduce_sum(tf.square(self.Y - predicted_Y))
        self.train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        feed_dict = {
            self.X: X,
            self.Y: Y
        }
        self.session.run(self.train_op, feed_dict=feed_dict)

    def predict(self, X):
        X = np.atleast_2d(X)
        feed_dict = {
            self.X: X
        }
        return self.session.run(self.predict_op, feed_dict=feed_dict)

def play_one_monte_carlo(env, value_model, policy_model, gamma):
    done = False

    total_rewards = 0
    actions = []
    states = []
    rewards = []

    reward = 0
    state = env.reset()
    # monti carlo means that we need to ask it
    while not done:
        # should we have all the previous action to calculate new policy
        action = policy_model.next_action(state)

        prev_state = state

        actions.append(action)
        rewards.append(reward)
        states.append(prev_state)

        state, reward, done, info = env.step(action)

        # very easy one trick to inforce it to learn
        if done:
            reward = -200
        else:
            total_rewards += reward

        # We do not update ANY! model when we play an Episode!

    action = policy_model.next_action(state)
    actions.append(action)
    rewards.append(reward)
    states.append(prev_state)
    advantages = []
    returns = []
    # All G are returns
    G = 0
    for state, reward in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - value_model.predict(state)[0])
        # the most tricky place! We mast take into account all the previous rewards
        G = reward + gamma * G

    advantages.reverse()
    returns.reverse()
    policy_model.partial_fit(states, actions, advantages)
    value_model.partial_fit(states, returns)

    return total_rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    dimensions_n = env.observation_space.shape[0]
    actions_n = env.action_space.n

    policy_model = PolicyModel(dimensions_n, actions_n, [10])
    value_model = ValueModel(dimensions_n, [10])

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    policy_model.set_session(session)
    value_model.set_session(session)

    N = 500
    gamma = 0.99
    total_rewards = np.empty(N)
    for i in range(N):
        reward = play_one_monte_carlo(env, value_model, policy_model, gamma)
        total_rewards[i] = reward
        if i % 100 == 0:
            print("Episode: ", i, " reward: ", reward)
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
