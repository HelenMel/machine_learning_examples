import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

from monte_carlo_random import random_action, play_game, SMALL_ENOUGH, GAMMA

LEARNING_RATE = 0.001

if __name__ == '__main__':
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    # init policy
    policy = {
        (2,0): 'U',
        (1,0): 'U',
        (0,0): 'R',
        (0,1): 'R',
        (0,2): 'R',
        (1,2): 'U',
        (2,1): 'L',
        (2,2): 'U',
        (2,3): 'L'
     }

    # init gradient descent. Randomize it to learn faster
    the = np.random.rand(4) / 2

    #for report
    deltas = []
    t = 1.0

    def state_to_features(s):
        x = s[0]; y = s[1]
        return np.array([x - 1, y - 1.5, x * y - 3, 1])

    # start training
    for time in range(20000):
        if time % 100 == 0:
            t += 1e-2
        alpha = LEARNING_RATE / t
        biggest_change = 0
        seen_states = set()
        # play game
        states_and_returns = play_game(grid, policy)

        # play game
        for s, G in states_and_returns:
            if s not in seen_states:
                old_the = the.copy()
                features = state_to_features(s)
                V_hat = the.dot(features)
                #gradient descent
                the = the + alpha*(G - V_hat) * features
                biggest_change = max(biggest_change, np.abs(old_the - the).sum())
                seen_states.add(s)
        deltas.append(biggest_change)

    # show result values
    plt.plot(deltas)
    plt.show()

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = the.dot(state_to_features(s))
        else:
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy")
    print_policy(policy,grid)