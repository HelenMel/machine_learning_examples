import tensorflow as tf
import carpol_q_learning

# this regressor merge models for all possible actions
class FullyConnectedNNRegressor:
    def __init__(self, dimensions, learning_rate):
        # remember variables state scopes that are required for
        # debugging purposes
        # TODO: define right variable scope
        #with tf.variable_scope('FullyConnectedQLearning', reuse=True):
        self.X = tf.placeholder(tf.float32, [None, dimensions], name='X')
        # expected output
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # how to backpropagate reward?
        self.reluHidden1 = tf.contrib.layers.fully_connected(self.X, 10)
        self.reluHidden2 = tf.contrib.layers.fully_connected(self.reluHidden1, 10)
        # fully connected layer is not the best choice here
        # convolutional + dropout would be great

        self.output = tf.contrib.layers.fully_connected(self.reluHidden2, 1, activation_fn=None)
        self.Y_hat = tf.reshape(tf.reduce_sum(self.output, axis=1), [-1])
        self.loss = tf.reduce_mean(tf.square(self.Y - self.Y_hat))
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # lets start a session
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        # I have to check that a_ could be fulfield with Y ([G])
        # and how to update targeQs
        feed = {self.X: X,
                self.Y: Y
        }
        loss, _ = self.session.run([self.loss, self.opt], feed_dict=feed)

        # not sure this is needed here
        # episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        # target_Qs[episode_ends] = (0, 0)

    def predict(self, X):
        feed = {self.X: X}
        return self.session.run(self.Y_hat, feed_dict=feed)

if __name__ == '__main__':
    carpol_q_learning.MySGDRegressor = FullyConnectedNNRegressor
    carpol_q_learning.main()