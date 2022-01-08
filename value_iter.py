from gridworld import GridWorld
from collections import defaultdict

from policy_eval import policy_eval
from policy_iter import get_greedy_policy


def argmax(d):
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def value_iter_onestep(env, gamma, V):
    # 一回分各状態における価値関数を更新
    # 更新差分
    delta = 0

    for state in env.states():
        action_values = []

        for action in env.actions():
            next_state = env.next_state(state, action)

            if next_state is not None:
                r = env.reward(state, action, next_state)
                value = r + gamma*V[next_state]
                action_values.append(value)

        if len(action_values) > 0:
            # 価値反復ではargmaxは関係ない
            new_value = max(action_values)
            delta = max(delta, abs(new_value - V[state]))
            V[state] = new_value

    return V, delta


def value_iter(env, gamma, threshold=1e-3, is_render=True):
    V = defaultdict(lambda: 0)
    while True:
        if is_render:
            env.render_v(V)

        V, delta = value_iter_onestep(env, gamma, V)
        if delta < threshold:
            break

    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    V = value_iter(env, gamma)

    pi = get_greedy_policy(V, env, gamma)
    env.render_v(V, pi)
