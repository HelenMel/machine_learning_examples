import gym
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from gym import wrappers
from mountain_car_q_learning import plot_running_avg, StateTransformer

# continious action space is unusual because we have to replace actions with its distribution
## We implement hidden layers themselfs! Yeeee!

class HiddenLayer:
    # bias is required only for hidden layers, not for an output layer that is usually softmax
    def __init__(self, input_n, output_n, activation=T.nnet.relu, use_bias=True, zeros=False):
        # we could not identify input matrix X size [None, input]
        # weights - create and randomize
        if zeros:
            Weights = np.zeros((input_n, output_n))
        else:
            Weights = np.random.randn(input_n, output_n)
        self.Weights = theano.shared(Weights)
        self.params = [self.Weights]
        self.use_bias = use_bias
        if use_bias:
            self.bias = theano.shared(np.zeros(output_n))
            self.params += [self.bias]
        self.activation = activation

    def forward(self, X):
        if self.use_bias:
            s = X.dot(self.Weights) + self.bias
        else:
            s = X.dot(self.Weights)
        return self.activation(s)

# action space is continious
class PolicyModel:
    # we talk about dimensions for state
    def __init__(self, state_transformer, dimensions, hidden_layer_sizes_mean=[], hidden_layer_sizes_variance=[]):
        # we have to save all params to recreate/copy Policy model later
        self.state_transformer = state_transformer
        self.dimensions = dimensions
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_variance = hidden_layer_sizes_variance


        # our model should predict aka parametrise normal Gaussian distribution
        ### model the mean
        self.mean_layers = []
        input_n = dimensions

        for output_n in hidden_layer_sizes_mean:
            layer = HiddenLayer(input_n, output_n)
            self.mean_layers.append(layer)
            input_n = output_n

        # mean is ounbounded, so activation function is identity
        layer = HiddenLayer(input_n, 1, activation= lambda  x: x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)

        ### model the variance
        self.variance_layers = []
        input_n = dimensions

        for output_n in hidden_layer_sizes_variance:
            layer = HiddenLayer(input_n, output_n)
            self.variance_layers.append(layer)
            input_n = output_n

        # we use softplus because values acre more than 0
        layer = HiddenLayer(input_n, 1, T.nnet.softplus, use_bias=False, zeros=False)
        self.variance_layers.append(layer)

        # TODO we have to colect the params. Why? 1:30
        params = []
        for layer in (self.mean_layers + self.variance_layers):
            params += layer.params
        caches = [theano.shared(np.ones_like(p.get_value())*0.1) for p in params]
        velocities = [theano.shared(p.get_value()*0) for p in params]
        self.params = params

        X = T.matrix('X')
        actions = T.vector('actions')
        advantages = T.vector('advantages')

        # lets crete a function to not repeat the same things for mean and variance
        def get_output(layers, X):
            out = X
            for layer in layers:
                out = layer.forward(out)
            return out.flatten()

        mean = get_output(self.mean_layers, X)
        # TODO? we need smoothing because it helps an exploration
        variance = get_output(self.variance_layers, X) + 0.00001 # smothing

        # pdf - probability density function
        # we have to find a log, not a pre pdf, because it is part of
        # the policy formula
        def log_pdf(points, mean, variance):
            # normal pdf is exp(-(points - mean) ** 2/ 2*variance) / sqrt(2 * pi * variance
            k1 = T.log(2 * np.pi * variance)
            k2 = (points - mean) ** 2 / variance
            return -0.5 * (k1 + k2)

        # because we have infinite amount of actions, we have to select
        # the most probable one
        Y = log_pdf(actions, mean, variance)
        # TODO: Why this formula?
        cost = - T.sum(advantages * Y + 0.1 * T.log(2 * np.pi*variance)) + 1.0 * mean.dot(Y)

        mu = 0.
        decay = 0.999
        learning_rate = 0.001

        grads = T.grad(cost, params)
        grads_update = [(p, p + v) for p, v, g in zip(params, velocities, grads)]
        cache_update = [(c, decay*c + (1 - decay)*g*g) for c, g in zip(caches, grads)]
        velocity_update = [(v, mu*v - learning_rate*g / T.sqrt(c)) for v, c, g in zip(velocities, caches, grads)]
        updates = cache_update + grads_update + velocity_update

        self.train_op = theano.function(
            inputs=[X, actions, advantages],
            updates=updates,
            allow_input_downcast=True
        )

        self.predict_op = theano.function(
            inputs=[X],
            outputs=[mean, variance],
            allow_input_downcast=True
        )

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.predict_op(self.state_transformer.trarnsform(X))

    # next action is the random action but that uses a current distribution
    def next_action(self, X):
        p = self.predict(X)
        # we only have one X, so we are interested in only first output
        mean = p[0][0]
        variance = p[1][0]
        action = np.random.randn()*np.sqrt(variance) + mean
        # our value should be in range [-1; 1]
        return min(max(action, -1), 1)

    def copy(self):
        self_copy = PolicyModel(self.state_transformer,
                                self.dimensions,
                                self.hidden_layer_sizes_mean,
                                self.hidden_layer_sizes_variance)
        self_copy.copy_state_from(self_copy)
        return self_copy

    def copy_state_from(self, other):
        for this_param, other_param in zip(self.params, other.params):
            this_param.set_value(other_param.get_value())

    # add some random noise to the parameters to start exploration
    def randomise_params(self):
        for p in self.params:
            value = p.get_value()
            noise = np.random.randn(*value.shape) / np.sqrt(value.shape[0]) * 5.0
            if np.random.random() < 0.1:
                p.set_value(noise)
            else:
                p.set_value(value + noise)

def play_one(env, policy_model, gamma):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = policy_model.next_action()
        state, reward, done, info = env.step(action)

        total_reward += reward
    return total_reward


def play_multiple_episodes(N_episodes, env, policy_model, gamma):
    total_rewards = np.empty(N_episodes)

    for i in range(N_episodes):
        total_rewards[i] = play_one(env, policy_model, gamma)

    avg_total = total_rewards.mean()
    return avg_total

def random_search(env, policy_model, gamma):
    total_rewards = []
    best_avg_total_reward = float('-inf')
    best_model = policy_model
    n_episodes_per_test = 3
    for t in range(100):
        temp_model = best_model.copy()
        temp_model.randomise_params()

        avg_result = play_multiple_episodes(n_episodes_per_test, env, temp_model, gamma)
        if avg_result > best_avg_total_reward:
            best_avg_total_reward = avg_result
            best_model = temp_model
        total_rewards.append(avg_result)

    return total_rewards, best_model

def main():
    env = gym.make('MountainCarContinious-v0')
    state_transformer = StateTransformer(env)
    dimensions = env.observation_space.shape[0]
    policy_model = PolicyModel(state_transformer, dimensions)
    gamma = 0.99

    total_rewards, best_policy_model = random_search(env, policy_model, gamma)

    best_avg_reward = play_multiple_episodes(100, env, best_policy_model, gamma)
    print("avg reward for BEST model: ", best_avg_reward)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

if __name__ == '__main__':
    main()
