import numpy as np
import tensorflow as tf
import carpol_q_learning

class SGDRegressor:
    def __init__(self, dimensions, learning_rate):
        self.learning_rate = 0.01

        # Tensor flow work similar to Teano it require to init basic functions
        # Beforehand and than just call following functions during
        # Predict or fit

        self.weights = tf.Variable(tf.random_normal(shape=(dimensions, 1)), name='weights')
        # placeholder is to init constant values during a fit that will cas as an input
        # None is used if you do not know in advance how many data(rows) you will have
        self.X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # The same thing as with theano - you should specify cost function to mesure error
        # but in case of SGD it will just sum up all the result to find gradient and
        # weights delta

        # predicted variable
        # instead of dot funcction we use matmul
        Y_hat = tf.reshape(tf.matmul(self.X, self.weights), [-1])
        delta = self.Y - Y_hat
        # squared error
        cost = tf.reduce_sum(delta * delta)

        # what is really required to update weights?
        # it looks like it ls just multiply consts to learrning rate, than update it
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        self.predict_op = Y_hat

        # in tensоrflow it is not enough to init variable
        # you also have to start a session to track training using session
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    # do we actually do a partial fit of retrain everything?
    # it looks like interactive session do the magic with partial fit
    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        a = self.session.run(self.predict_op, feed_dict={self.X: X})
        print(a.shape)
        return a

if __name__ == '__main__':
    carpol_q_learning.MySGDRegressor = SGDRegressor
    carpol_q_learning.main()