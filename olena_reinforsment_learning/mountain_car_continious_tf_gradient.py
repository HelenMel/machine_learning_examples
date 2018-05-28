import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mountain_car_q_learning import StateTransformer, plot_cost_to_go, plot_running_avg

class HiddenLayer:
    def __init__(self, n_input, n_output, activation=tf.nn.tanh, use_bias=True, zeros=False):
        # initialize weights
        if zeros:
            W = np.zeros((n_input, n_output), dtype=np.float32)
        else:
            W = tf.random_normal(shape=(n_input, n_output)) * np.sqrt(2. / n_input, dtype=np.float32)
        self.Weights = tf.Variable(W)
        self.use_bias = use_bias
        self.activation = activation

        if use_bias:
            self.b = tf.Variable(np.zeros(n_output).astype(np.float32))

    def forward(self, X):
        if self.use_bias:
            r = tf.matmul(X, self.Weights) + self.b
        else:
            r = tf.matmul(X, self.Weights)
        return self.activation(r)

class PolicyModel:
    def __init__(self, dimensions, state_transformer, hidden_layers_size_mean=[], hidden_layers_size_var=[]):
        # we don't have to copy everything because our next move is not random
        self.state_transformer = state_transformer
        # mean model
        mean_layers = []
        n_input = dimensions
        for n_output in hidden_layers_size_mean:
            mean_layers.append(HiddenLayer(n_input, n_output))
            n_input = n_output
        layer = HiddenLayer(n_input, 1, lambda x: x, use_bias=False, zeros=True)
        mean_layers.append(layer)

        # variance model
        variance_layers = []
        n_input = dimensions
        for n_output in hidden_layers_size_var:
            variance_layers.append(HiddenLayer(n_input, n_output))
            n_input = n_output

        layer = HiddenLayer(n_input, 1, tf.nn.softplus, use_bias=False, zeros=False)
        variance_layers.append(layer)

        # initialize required variables
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        # X represent a state
        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')


        # create a prediction function
        def run_layers(layers, X):
            Z = X
            for layer in layers:
                Z = layer.forward(Z)
            return Z

        mean = tf.reshape(run_layers(mean_layers, self.X), [-1])
        variance = tf.reshape(run_layers(variance_layers, self.X) + 0.00001, [-1])

        # now, when mean and variance predicted use
        predicted_distribution = tf.contrib.distributions.Normal(mean, variance)
        predicted_action = predicted_distribution.sample()
        self.predicted_op = tf.clip_by_value(predicted_action, -1, 1)

        # create a cost/train function
        # actually we want to maximize advantage per actions
        log_probs = predicted_distribution.log_prob(self.actions)
        # we also try to maximize entropy to increase variance
        cost = - tf.reduce_sum(self.advantages * log_probs + 0.1 * predicted_distribution.entropy())
        learning_rate = 0.001
        self.train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, actions, advantages, states):
        states = np.atleast_2d(states)
        X = self.state_transformer.transform(states)
        # we need X to find a new distribution
        actions = np.atleast_1d(actions)
        advantages =np.atleast_1d(advantages)
        feed_dict = {
            self.actions: actions,
            self.advantages: advantages,
            self.X: X
        }
        self.session.run(self.train_op, feed_dict=feed_dict)

    def predict(self, states):
        states = np.atleast_2d(states)
        X = self.state_transformer.transform(states)
        return self.session.run(self.predicted_op, feed_dict={self.X: X})

    def next_action(self, states):
        # to find a next action we should
        return self.predict(states)[0]

# we need a value model here because:
# value model helps to identify advantage
# to learn smth after each episode
class ValueModel:
    def __init__(self, dimensions, state_transformer, hidden_layer_sizes=[]):
        self.state_transformer = state_transformer
        # TODO: we save a costs after each fit session because...
        self.costs = []

        self.layers = []
        n_input = dimensions
        for n_output in hidden_layer_sizes:
            self.layers.append(HiddenLayer(n_input, n_output))
            n_input = n_output

        # TODO: why it is just an identity
        self.layers.append(HiddenLayer(n_input, 1, lambda x:x))

        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        # this is observable variable we try to minimize difference with
        self.Vtarget = tf.placeholder(tf.float32, shape=(None,), name='Vtarget')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        # we have to reshape, because model always returns an array
        Vpredicted = tf.reshape(Z, [-1])

        # train operation
        cost = tf.reduce_sum(tf.square(self.Vtarget - Vpredicted))
        self.cost = cost
        self.train_op = tf.train.AdamOptimizer(0.1).minimize(cost)

        # prediction operation
        self.predict_op = Vpredicted

    def set_session(self, session):
        self.session = session

    def partial_fit(self, state, V):
        state = np.atleast_2d(state)
        X = self.state_transformer.transform(state)
        V = np.atleast_1d(V)
        self.session.run(self.train_op, feed_dict={ self.X: X, self.Vtarget: V })
        cost = self.session.run(self.cost, feed_dict={ self.X: X, self.Vtarget: V } )
        self.costs.append(cost)

    def predict(self, state):
        state = np.atleast_2d(state)
        X = self.state_transformer.transform(state)
        return self.session.run(self.predict_op, feed_dict={ self.X: X})

def run_once(env, policy_model, value_model, gamma):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = policy_model.next_action(state)
        prev_state = state
        state, reward, done, info = env.step([action])

        total_reward += reward

        Vnext = value_model.predict(state)
        G = reward + gamma * Vnext
        advantage = G - value_model.predict(prev_state)
        # it seems like we only calculate an advantage for the closest model
        policy_model.partial_fit(action, advantage, state)
        value_model.partial_fit(state, G)

    return total_reward

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    state_transformer = StateTransformer(env)
    dimensions = state_transformer.dimensions

    policy_model = PolicyModel(dimensions, state_transformer, [], [])
    value_model = ValueModel(dimensions, state_transformer, [])

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    policy_model.set_session(session)
    value_model.set_session(session)
    gamma = 0.95


    N_episodes = 50
    total_rewards = np.empty(N_episodes)
    # we dont use it. WTF?
    costs = np.empty(N_episodes)

    for i in range(N_episodes):
        reward = run_once(env, policy_model, value_model, gamma)
        total_rewards[i] = reward
        if i % 3 == 0:
            print("total reward: ", reward)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
    plot_cost_to_go(env, value_model)

