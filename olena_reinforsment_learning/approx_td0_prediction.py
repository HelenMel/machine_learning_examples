import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values,  print_policy

GAMMA = 0.9
ALPHA = 0.1
ALL_ACTIONS = ('U', 'D', 'L', 'R')
LEARNING_RATE = 0.001

class Model:
    def __init__(self):
        # we have 4 dimensions: x, y anf its combination
        self.theta = np.random.rand(4) / 2

    def state_to_features(self, s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] -3, 1])

    def predict(self, s):
        features = self.state_to_features(s)
        return self.theta.dot(features)

    def grad(self, s):
        return self.state_to_features(s)

def random_action(a, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_ACTIONS)

def play_game(grid, policy):
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards

if __name__ == '__main__':
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U'
    }

    V = {}

    model = Model()
    deltas = []

    t = 1.0
    for time in range(20000):
        if time % 10:
            t += 1e-2
        alpha = ALPHA / t

        biggest_change = 0

        states_and_rewards = play_game(grid, policy)
        for i in range(len(states_and_rewards) - 1):
            s1, r1 = states_and_rewards[i]
            s2, r2 = states_and_rewards[i + 1]
            old_the = model.theta.copy()

            if grid.is_terminal(s2):
                target = r2
            else:
                target = r2 + GAMMA * model.predict(s2)
            model.theta = model.theta + alpha * (target -  model.predict(s1)) *  model.grad(s1)

            biggest_change = max(biggest_change, np.abs((old_the - model.theta).sum()))
        deltas.append(biggest_change)


    plt.plot(deltas)
    plt.show()

    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy")
    print_policy(policy, grid)