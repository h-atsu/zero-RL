import numpy as np
from gridworld import GridWorld
from collections import defaultdict
from mcpolicy import McAgent, greedy_action_probs


class McOffPolicyAgent(McAgent):
    def __init__(self):
        super().__init__()
        self.b = self.pi.copy()

    def get_action(self, state):
        # 行動を行う際の方策はb
        ps = self.b[state]
        actions, probs = list(ps.keys()), list(ps.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        self.experience.append((state, action, reward))

    def update(self):
        g = 0
        # 重点サンプリングの際の重み
        rho = 1
        for data in reversed(self.experience):
            state, action, reward = data
            # Q関数は(state,action)を受け取って価値を返す
            key = (state, action)

            g = self.gamma * rho * g + reward
            self.Q[key] += (g - self.Q[key]) * self.alpha
            rho *= self.pi[state][action] / self.b[state][action]

            self.pi[state] = greedy_action_probs(self.Q, state)
            self.b[state] = greedy_action_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

episodes = 10000

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
