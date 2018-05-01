## In this file I'm try to implement Q-learning using
## stohastic gradient descent and RBFSampler as a kernels
## we don't use Q table anymore but it is still Q learning
## because we learn online, each step use a prediction for a next step
## and because we have an exploration phase to encourage learning

import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

# sklearn has its own SGDRegressor but here is custom implementation for a learning purpose
class MySGDRegressor:
    def __init__(self, dimensions, learning_rate):
        self.weights = np.random.random(dimensions) / np.sqrt(dimensions)
        self.lr = learning_rate

    def fit(self, X, Y):
        pass

    def partial_fit(self, X, Y):
        observed = Y
        predicted = X.dot(self.weights)
        self.weights += self.lr * (observed - predicted).dot(X)

    def predict(self, X):
        return X.dot(self.weights)

class StateTransformer:
    def __init__(self, env):
        # this doesn't make sense to get a samples for a kernels from enviroment random variables
        # because they return some states that are almost impossible to get
        N_examples = 20000
        N_state_dimensions = 4
        # range of examples -2 to 2
        examples = np.random.random((N_examples, N_state_dimensions)) * 4 - 2
        scaler = StandardScaler()
        scaler.fit(examples)
        _kernels = FeatureUnion([
            ("rbf1", RBFSampler())
        ])

        # we need number of dimensions to create MySGDRegressor
        self.dimensions