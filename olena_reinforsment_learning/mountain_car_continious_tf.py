import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mountain_car_q_learning import StateTransformer


# This is almost the same layer from policy gradient
# the difference is that we assign weights to zero and save a
# params to copy layer later
class HiddenLayer:
    def __init__(self, n_input, n_output, activation=tf.nn.tanh, use_bias=True, zeros=False):
        self.use_bias = use_bias
        self.activation = activation
        if zeros:
            z = np.zeros((n_input, n_output)).astype(np.float32)
            self.Weights = tf.Variable(z)
        else:
            self.Weights = tf.Variable(tf.random_normal(shape=(n_input, n_output)))

        self.params = [self.Weights]
        if use_bias:
            self.b = tf.Variable(np.zeros(n_output).astype(np.float32))
            self.params.append(self.b)

    def forward(self, X):
        a = tf.matmul(X, self.Weights)
        if self.use_bias:
            a = a + self.b
        return self.activation(a)

# here is TF implementation of hill climbing policy - random assignment of params
# in this case we don't need an value model
class PolicyModel:
    def __init__(self, dimensions, state_transformer, hidden_layers_size_mean=[], hidden_layers_size_var=[]):
        # save params to copy it later
        self.dimensions = dimensions
        self.state_transformer = state_transformer
        self.hidden_layer_size_mean = hidden_layers_size_mean
        self.hidden_layer_size_var = hidden_layers_size_var


        # mean model
        self.mean_layers = []
        n_input = dimensions
        for n_output in hidden_layers_size_mean:
            layer = HiddenLayer(n_input, n_output)
            self.mean_layers.append(layer)
            n_input = n_output
        # activation function is identity because mean is unbounded
        layer = HiddenLayer(n_input, 1, lambda x: x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)

        # variance model
        self.variance_layers = []
        n_input = dimensions
        for n_output in hidden_layers_size_var:
            layer = HiddenLayer(n_input, n_output)
            self.variance_layers.append(layer)
            n_input = n_output

        # variance should be more than zero
        layer = HiddenLayer(n_input, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.variance_layers.append(layer)

        # collect params to make a COPY of it later
        self.params = []
        for layer in (self.mean_layers + self.variance_layers):
            self.params += layer.params

        # input
        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantage = tf.placeholder(tf.float32, shape=(None,), name='advantage')


        # find a cost
        def run_net(layers, X):
            result = X
            for layer in layers:
                result = layer.forward(result)
            return tf.reshape(result, [-1])

        predicted_mean = run_net(self.mean_layers, self.X)
        predicted_variance = run_net(self.variance_layers, self.X) + 0.00001

        predicted_distribution = tf.contrib.distributions.Normal(predicted_mean, predicted_variance)

        # bound predicted action to range [-1, 1]
        self.predict_op = tf.clip_by_value(predicted_distribution.sample(), -1, 1)


        # during the training we should output both mean and variance
        # TODO: combine hill climbing with optimization
        # log_probability =  predicted_distribution.log_prob(self.actions)
        # self.cost = -tf.reduce_sum(self.advantage * log_probability + 0.1* tf.log(2*np.pi*predicted_variance))
        #
        # learning_rate = 0.01
        #
        # self.train_op = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)

    def set_session(self, session):
        self.session = session

    # we cannot use global variables initializer, because it will reset all the variables we have used before
    # but we have to keep best model
    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)


    # not use it here
    # def partial_fit(self, X, actions, advantages):
    #     X = np.atleast_2d(X)
    #     X = self.state_transformer.transform(X)
    #
    #     actions = np.atleast_1d(actions)
    #     advantages = np.atleast_1d(advantages)
    #     feed_dict = {
    #         self.X: X,
    #         self.actions: actions,
    #         self.advantage: advantages
    #     }
    #     self.session.run(self.train_op, feed_dict=feed_dict)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.state_transformer.transform(X)
        feed_dict = { self.X: X }
        return self.session.run(self.predict_op, feed_dict=feed_dict)

    # next action is actually a random action according to a current distribution
    def next_action(self, X):
        action = self.predict(X)[0]
        return action

    def copy(self):
        new_model = PolicyModel(self.dimensions,
                                self.state_transformer,
                                self.hidden_layer_size_mean,
                                self.hidden_layer_size_var)
        new_model.set_session(self.session)
        new_model.init_vars()
        new_model.copy_params(self)
        return new_model

    #
    def copy_params(self, other):
        operations = []
        my_params = self.params
        other_params = other.params
        for my_p, other_p in zip(my_params, other_params):
            actual = self.session.run(other_p)
            op = my_p.assign(actual)
            operations.append(op)
        self.session.run(operations)

    # according to a hill climbing we have to randomise policy parameters each time
    def randomize_params(self):
        # according to the tensorflow we are not only run an operation
        # but also run it in sessions
        operations = []
        for p in self.params:
            param_value = self.session.run(p)
            # add noise to the parameter value
            noise = np.random.randn(*param_value.shape) / np.sqrt(param_value.shape[0]) * 5.0

            if np.random.random () < 0.1:
                # this helps us not to stuck on a local minimum
                operation = p.assign(noise)
            else:
                operation = p.assign(param_value + noise)
            operations.append(operation)
        self.session.run(operations)


# run one episode
def run_once(env, policy_model):
    total_reward = 0
    done = False
    state = env.reset()

    while not done:
        action = policy_model.next_action(state)
        state, reward, done, info = env.step([action])
        total_reward += reward

    return total_reward

# try one model
def run_multiple_episodes(env, n_episodes, policy_model):
    total_rewards = np.empty(n_episodes)

    for i in range(n_episodes):
        total_rewards[i] = run_once(env, policy_model)

    avg_reward = total_rewards.mean()
    return avg_reward

# assign and try different models with random assignments
def play_random_policy(N_trials, env, policy_model):
    best_model = policy_model
    best_reward = float('-inf')
    total_rewards = []

    for t in range(N_trials):
        tmp_model = best_model.copy()
        tmp_model.randomize_params()
        avg_reward = run_multiple_episodes(env, 3, tmp_model)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = tmp_model
        total_rewards.append(avg_reward)

    return total_rewards, best_model

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    st = StateTransformer(env)
    dimensions = st.dimensions
    policy_model = PolicyModel(dimensions, st)
    session = tf.InteractiveSession()
    policy_model.set_session(session)
    policy_model.init_vars()

    total_rewards, best_model = play_random_policy(100, env, policy_model)
    print("max reward:", np.max(total_rewards))

    # run final model
    best_model_reward = run_multiple_episodes(env, 100, best_model)
    print("Best model reward: ", best_model_reward)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()





