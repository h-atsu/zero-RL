import numpy as np
from gridworld import GridWorld
from collections import defaultdict


def greedy_action_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    base_prob = epsilon / action_size  # ε/4の一様分布で初期化
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1.0 - epsilon)
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.1
        self.action_state = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.experience = []

    def get_action(self, state):
        ps = self.pi[state]
        actions, probs = list(ps.keys()), list(ps.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        self.experience.append((state, action, reward))

    def reset(self):
        self.experience.clear()

    def update(self):
        g = 0
        for data in reversed(self.experience):
            state, action, reward = data
            g = self.gamma * g + reward
            # Q関数は(state,action)を受け取って価値を返す
            key = (state, action)
            self.Q[key] += (g - self.Q[key]) * self.alpha
            self.pi[state] = greedy_action_probs(self.Q, state, self.epsilon)


if __name__ == '__main__':
    env = GridWorld()
    agent = McAgent()

    episodes = 5000

    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break

            state = next_state

    env.render_q(agent.Q)
