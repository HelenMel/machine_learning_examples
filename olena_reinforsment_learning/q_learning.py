import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values
from monte_carlo_es import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)
    print("rewards:")
    print_values(grid.rewards, grid)

    Q = {}
    update_counts_sa = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            update_counts_sa[s][a] = 1.0

    update_counts = {}

    t = 1.0
    deltas = []
    for time in range(10000):
        if time % 100 == 0:
            time += 1e-2
        if time % 2000 == 0:
            print("time {0}", time)

        #set initial state
        s1 = (2, 0)
        grid.set_state(s1)
        #first action
        a1 = max_dict(Q[s1])[0]
        biggest_change = 0
        while not grid.game_over():
            a1 = random_action(a1, eps=0.5 / t)
            r = grid.move(a1)
            s2 = grid.current_state()

            alpha = ALPHA / update_counts_sa[s1][a1]
            update_counts_sa[s1][a1] += 0.005

            old_q = Q[s1][a1]
            a2, max_q_2 = max_dict(Q[s2])

            Q[s1][a1] = Q[s1][a1] + alpha * (r + GAMMA * max_q_2 - Q[s1][a1])
            biggest_change = max(biggest_change, np.abs(Q[s1][a1] - old_q))

            update_counts[s] = update_counts.get(s, 0) + 1
            a1 = a2
            s1 = s2
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # determine the policy from Q*
    # find V* from Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what's the proportion of time we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)





