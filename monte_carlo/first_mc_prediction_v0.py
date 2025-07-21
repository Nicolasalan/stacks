import numpy as np
from typing import Tuple

INITIAL_STATE = (1, 0)
STATE = (1, 0)

states = [(i, j) for i in (0, 1) for j in (0, 1)]
actions = ["top", "down", "left", "right"]
probs = [0.25, 0.25, 0.25, 0.25]
Q_TABLE = {}

for s in states:
    for a in actions:
        Q_TABLE[(s, a)] = [0.0]

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
    for i in range(ep):
        step = 0
        first_visit = dict()
        rewards = []
        reward = 0
        epsodio = []
        while True:
            action = np.random.choice(actions, p=probs)
            if i >= 90:
                # greedy policy
                action = np.str_(max(actions, key=lambda a: Q_TABLE.get((STATE, a), [0])[0]))

            new_state, reward, done = env(STATE, action)
            rewards.append(reward)
            pair = (STATE, action.item())
            if pair not in first_visit:
                first_visit[pair] = step
            step += 1
            epsodio.append((new_state, action, reward))
            STATE = new_state

            print(f"Ep {i}: state: {STATE}; reward: {reward}; action: {action}")
            if done == True:
                # policy iteration
                G = 0
                for (state, action), t in first_visit.items():
                    for idx, _ in enumerate(rewards):
                        if t == idx:
                            G = sum(rewards[t:])
                            state_action = (state, action)
                            Q_TABLE[state_action].append(G)

                for (state, action), values in Q_TABLE.items():
                    med = sum(values) / len(values)
                    Q_TABLE[(state, action)] = [med]
                rewards.clear()
                first_visit.clear()
                STATE = INITIAL_STATE
                break
