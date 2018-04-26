import numpy as np
import matplotlib.pyplot as plot
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values,  print_policy

GAMMA = 0.9
ALPHA = 0.1
ALL_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_ACTIONS)

def play_game(grid, policy):
    s = (2,0)
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
    print("reward")
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

    # init
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    # calculate V table
    for it in range(1000):
        states_and_rewards = play_game(grid, policy)
        for time in range(len(states_and_rewards) - 1):
            s1, r1 = states_and_rewards[time]
            s2, r2 = states_and_rewards[time + 1]
            V[s1] = V[s1] + ALPHA * (r2 + GAMMA * V[s2] - V[s1])

    print("values:")
    print_values(V, grid)
    print("policy")
    print_policy(policy, grid)


