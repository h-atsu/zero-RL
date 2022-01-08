from gridworld import GridWorld
from collections import defaultdict

from policy_eval import policy_eval


def argmax(d):
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def get_greedy_policy(V, env, gamma):
    """
    現在の状態価値関数を受け取り，argmaxとなる行動を方策として返す
    """
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            value = 0

            if next_state is not None:
                r = env.reward(state, action, next_state)
                value += r + gamma*V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)  # get key value
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs

    return pi


def policy_iter(env, gamma, threshold=1e-3, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        # 現在の暫定方策の元で状態価値関数を評価
        V = policy_eval(pi, V, env, gamma, threshold)
        # 現在の暫定方策を状態価値関数を用いて改善する
        new_pi = get_greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        # 決定論的方策のため整数値として一致するかしないかを確認
        if new_pi == pi:
            break

        pi = new_pi

    return pi


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
