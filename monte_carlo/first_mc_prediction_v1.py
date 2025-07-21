import numpy as np
from typing import Tuple

N, Q, returns_sum = {}, {}, {}

def env(state, action: str) -> Tuple[tuple, float, bool]:

    grid = [[-1, -1],
            [-1, 10]]
    done = False

    new_pos = (0, 0)

    row, col = state
    if action == "top":
        new_pos = (row-1, col)
        if new_pos[0] < 0:
            new_pos = state
    elif action == "down":
        new_pos = (row+1, col)
        if new_pos[0] > 1:
            new_pos = state
    elif action == "left":
        new_pos = (row, col-1)
        if new_pos[1] < 0:
            new_pos = state
    elif action == "right":
        # hard
        if state == (1, 0):
            new_pos = state
        else:
            new_pos = (row, col+1)
            if new_pos[1] > 1:
                new_pos = state

    new_state = new_pos
    reward = grid[new_state[0]][new_state[1]]

    if reward == 10:
        done = True

    return new_state, reward, done

if __name__ == "__main__":
    ep = 100
    state = (1, 0)
    actions = ["top", "down", "left", "right"]
    for i in range(ep):
        eps = []
        done  = False
        while not done:
            action = np.random.choice(actions, p=[0.25, 0.25, 0.25, 0.25])
            if i >= 90:
                # greedy policy
                action = np.str_(max(actions, key=lambda a: Q.get((state, a), 0)))

            new_state, reward, done = env(state, action)
            eps.append((new_state, action.item(), reward))
            state = new_state
            print(f"Ep {i}: state: {state}; reward: {reward}; action: {action}")

        G, see = 0.0, set()
        for (s, a, r) in reversed(eps):
            G += r
            if  (s, a) in see:
                continue
            see.add((s, a))
            N[(s, a)] = N.get((a, s), 0.0) + 1
            returns_sum[(s, a)] = returns_sum.get((s, a), 0.0) + G
            Q[(s, a)] = returns_sum[(s, a)]/N[(s, a)]

        print("Q table: ", len(Q))
