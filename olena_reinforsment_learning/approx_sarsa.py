import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values
from monte_carlo_es import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
LEARNING_RATE = 0.001

SA2IDX = {}
IDX = 0


class SARSAModel:
    def __init__(self):
        self.theta = np.random.rand(25) / np.sqrt(25)

    def sa2x(self, s, a):
        return np.array([
      s[0] - 1              if a == 'U' else 0,
      s[1] - 1.5            if a == 'U' else 0,
      (s[0]*s[1] - 3)/3     if a == 'U' else 0,
      (s[0]*s[0] - 2)/2     if a == 'U' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
      1                     if a == 'U' else 0,
      s[0] - 1              if a == 'D' else 0,
      s[1] - 1.5            if a == 'D' else 0,
      (s[0]*s[1] - 3)/3     if a == 'D' else 0,
      (s[0]*s[0] - 2)/2     if a == 'D' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
      1                     if a == 'D' else 0,
      s[0] - 1              if a == 'L' else 0,
      s[1] - 1.5            if a == 'L' else 0,
      (s[0]*s[1] - 3)/3     if a == 'L' else 0,
      (s[0]*s[0] - 2)/2     if a == 'L' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
      1                     if a == 'L' else 0,
      s[0] - 1              if a == 'R' else 0,
      s[1] - 1.5            if a == 'R' else 0,
      (s[0]*s[1] - 3)/3     if a == 'R' else 0,
      (s[0]*s[0] - 2)/2     if a == 'R' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
      1                     if a == 'R' else 0,
      1
    ])

    def predict(self, s, a):
        features = self.sa2x(s, a)
        return self.theta.dot(features)

    def grad(self, s, a):
        return self.sa2x(s, a)


def all_actions(s, model):
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs

if __name__=="__main__":
    grid= negative_grid(step_cost=-0.1)

    print("reward")
    print_values(grid.rewards, grid)

    #init approximated result table
    states = grid.all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1

    t = 1.0
    t2 = 1.0
    deltas = []
    model = SARSAModel()
    for time in range(20000):
        if time % 100 == 0:
            t += 0.01
            t2 += 0.01
        if time % 2000 == 0:
            print("time: {0}", time)
        alpha = ALPHA / t2
        s1 = (2,0)
        grid.set_state(s1)
        a1 = max_dict(all_actions(s1, model))[0]
        a1 = random_action(a1, eps=0.5/t)
        biggest_change = 0

        while not grid.game_over():
            r = grid.move(a1)
            s2 = grid.current_state()

            old_theta = model.theta.copy()

            if grid.is_terminal(s2):
                # grad is the same as features = model.sa2x(s1, a1)
                model.theta += alpha * (r - model.predict(s1, a1)) * model.grad(s1, a1)
            else:
                a2 = max_dict(all_actions(s2, model))[0]
                a2 = random_action(a2, eps=0.5/t)

                model.theta += alpha * (r + GAMMA * model.predict(s2, a2) - model.predict(s1, a1)) * model.grad(s1, a1)
                s1 = s2
                a1 = a2
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # we don't use Q table, at least during training processes.
    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = all_actions(s, model)
        Q[s] = Qs
        a, max_result = max_dict(Qs)
        policy[s] = a
        V[s] = max_result
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
