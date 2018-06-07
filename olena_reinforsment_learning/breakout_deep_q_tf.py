import gym
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
import random

from carpol_q_learning import plot_running_avg

IMAGE_W = 80
IMAGE_H = 80
N_FRAMES = 4 # amount of frames we process during ONE STEP

# this time we dont create a hidden layer class
# because we uses build it tensorflow layers

def downsample(img):
    B = img[35: 195]
    B = B.mean(axis=2)
    B = B / 255.0
    return imresize(B, size=(IMAGE_H, IMAGE_W), interp='nearest')


class DeepQNet:
    def __init__(self,
                 scope_name,
                 n_output,
                 conv_layer_sizes,
                 full_layer_sizes,
                 gamma,
                 max_experiences=10000, # maximum memory buffer size
                 min_experiences=100,
                 batch_size=32
                 ):
        # each layer should be created with unique prefix
        self.n_output = n_output
        self.scope_name = scope_name
        self.gamma = gamma

        with tf.variable_scope(scope_name):
            # create variable scope
            self.X = tf.placeholder(tf.float32, shape=(None, N_FRAMES ,IMAGE_H, IMAGE_W), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

            r = self.X
            # fix initial sample order. It should be: #samples, height, width, depth(frames)
            r = tf.transpose(r, [0, 2, 3, 1])
            for n_output_filters, filter_size, pool_size in conv_layer_sizes:
                # create a layer and the output will be in a last layer
                r = tf.contrib.layers.conv2d(r,
                                             n_output_filters,
                                             filter_size,
                                             pool_size,
                                             activation_fn=tf.nn.relu)
            # we reduce width and height for each conv layer, so it almost flat.
            # but we need flatted it explicitly to reduce depth, width and height that still left
            r = tf.contrib.layers.flatten(r)
            # now we have to connect it with a full size layers
            for output_size in full_layer_sizes:
                r = tf.contrib.layers.fully_connected(r, output_size)

            self.predict_op = tf.contrib.layers.fully_connected(r, n_output)

            selected_actions_gain = tf.reduce_sum(
                self.predict_op * tf.one_hot(self.actions, n_output),
                reduction_indices=[1]
            )

            cost = tf.reduce_sum(tf.square(selected_actions_gain - self.G))
            self.train_op = tf.train.RMSPropOptimizer(0.0025, decay=0.99, epsilon=1e-3).minimize(cost)

        self.replay_memory = []
        self.max_experience = max_experiences
        self.min_experience = min_experiences
        self.batch_size = batch_size

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        this_params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope_name)]
        other_params = [t for t in tf.trainable_variables() if t.name.startswith(other.scope_name)]
        # we should be sure, that we zip the same params,
        # that is why we have to sort them first
        this_params = sorted(this_params, key=lambda x: x.name)
        other_params = sorted(other_params, key=lambda x: x.name)

        ops = []
        for this_p, other_p in zip(this_params, other_params):
            other_value = self.session.run(other_p)
            ops.append(this_p.assign(other_value))

        self.session.run(ops)

    def predict(self, X):
        # if X.shape != (32, 4, 80, 80):
        #     print("Error shape ", X.shape)
        #     assert (False)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, stable_model):
        if len(self.replay_memory) < self.min_experience:
            return

        # we have to select random indexes
        sample = random.sample(self.replay_memory, self.batch_size)
        # replay memory is stored a bit differently, not like in a carpol_deep_q
        # use zip to connect tuple items together
        # use map to array to separate items
        states, actions, rewards, next_states = map(np.array, zip(*sample))

        next_Q = np.amax(stable_model.predict(next_states), axis=1)
        G = [r + self.gamma * q for r, q in zip(rewards, next_Q)]

        self.session.run(self.train_op, feed_dict={
            self.X: states,
            self.G: G,
            self.actions: actions
        })

    def update_memory(self, state_before, action, reward, state_after):
        if len(self.replay_memory) >= self.max_experience:
            self.replay_memory.pop(0)
        self.replay_memory.append((state_before, action, reward, state_after))

    def next_action(self, X, eps):
        if np.random.random() < eps:
            return np.random.choice(self.n_output)
        else:
            return np.argmax(self.predict(np.array([X]))[0])

def update_state_memory(state_memory, state):
    simplified_state = downsample(state)
    state_memory.append(simplified_state)
    if len(state_memory) > 4:
        state_memory.pop(0)

def run_once(env, model, stable_model, eps, eps_step, update_period):
    state = env.reset()
    total_rewards = 0
    done = False
    i = 0
    n_output = env.action_space.n

    prev_state_memory = []
    state_memory = []
    update_state_memory(state_memory, state)
    while not done and i < 2000:
        if len(state_memory) < N_FRAMES:
            # we have to use at least 4 frames to create X
            # that we use in a prediction, and next step
            action = np.random.choice(n_output)
        else:
            action = model.next_action(state_memory, eps)

        # we could use an update_state_memory here, but we don' need downsampling
        prev_state_memory.append(state_memory[-1])
        if len(prev_state_memory) > 4:
            prev_state_memory.pop(0)

        state, reward, done, info = env.step(action)

        update_state_memory(state_memory, state)
        total_rewards += reward
        if done:
            reward = -200

        if len(prev_state_memory) == 4 and len(state_memory) == 4:
            model.update_memory(prev_state_memory, action, reward, state_memory)
            model.train(stable_model)

        i += 1

        # in this version we should update eps faster, during one session
        # but it should never be less than 0.1
        eps = max(eps - eps_step, 0.1)
        if i % update_period == 0:
            stable_model.copy_from(model)

    return total_rewards, eps


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    gamma = 0.99
    update_period = 1000

    n_output = env.action_space.n
    conv_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_sizes = [512]

    model = DeepQNet("main", n_output, conv_sizes, hidden_sizes, gamma)
    stable_model = DeepQNet("stable", n_output, conv_sizes, hidden_sizes, gamma)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    model.set_session(session)
    stable_model.set_session(session)

    eps = 1.0
    eps_min = 0.1
    eps_step = (eps - eps_min) / 500000

    N_trials = 10000
    total_rewards = np.empty(N_trials)
    for t in range(N_trials):
        print("try: ", t)
        total_reward, _ = run_once(env, model, stable_model, eps, eps_step, 20)
        total_rewards[t] = total_reward
        if t % 100 == 0:
            print("episode:", t, "total reward:", total_reward)

    print("total steps:", total_rewards.sum())
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
