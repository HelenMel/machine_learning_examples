
import numpy as np
import theano
import theano.tensor as T

# Stohastic Gradient descent regressor
class SGDRegressor:
    def __init__(self, dimensions):
        weights = np.random.randn(dimensions) / np.sqrt(dimensions)
        self.weights = theano.shared(weights)
        self.learning_rate = 0.01
        X = T.matrix('X')
        Y = T.vector('Y')
        # predicted variables
        Y_hat = X.dot(self.weights)
        delta = Y - Y_hat
        # squared error
        cost = delta.dot(delta)
        # gradient could be found completely manually using
        # math wormula that make a gradient based on difference between predicted and real value Y
        # in teano this action should be initialized at first
        grad = T.grad(cost, self.weights)
        updates = [(self.weights, self.weights - self.learning_rate * grad)]

        # we initialize functions here, so during a learning process theano knows what to call
        self.train_op = theano.function(
            inputs=[X, Y],
            updates=updates
        )

        self.predict_op = theano.function(
            inputs=[X],
            outputs=Y_hat
        )

    def partial_fit(self, X, Y):
        self.train_op(X, Y)

    def predict(self, X):
        return self.predict_op(X)