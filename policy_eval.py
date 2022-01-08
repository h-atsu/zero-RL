from gridworld import GridWorld
from collections import defaultdict


def eval_onestep(pi, V, env, gamma=0.9):
    delta = 0

    for state in env.states():
        action_probs = pi[state]
        new_v = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            if next_state is not None:
                r = env.reward(state, action, next_state)
                new_v += action_prob * (r + gamma * V[next_state])

        delta = max(delta, abs(V[state] - new_v))
        V[state] = new_v

    return V, delta


def policy_eval(pi, V, env, gamma, threshold=1e-3):
    """
    evaluate policy by using DP
    """
    while True:
        V, delta = eval_onestep(pi, V, env, gamma)
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
