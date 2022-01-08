import numpy as np
import matplotlib.pyplot as plt


class NonstatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        reward = rate > np.random.rand()
        # add noise
        self.rates += 0.1*np.random.randn(self.arms)
        return int(reward)


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        a, r = action, reward
        self.ns[a] += 1
        self.qs[a] += (r - self.qs[a]) / self.ns[a]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.qs))
        return np.argmax(self.qs)


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        a, r = action, reward
        self.qs[a] += (r - self.qs[a]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.qs))
        return np.argmax(self.qs)


trials = 2000  # number of experiments trials
steps = 1000
policy = ['avg', 'alpha']
epsilon = 0.1
all_rates = np.zeros((trials, steps))


for pol in policy:
    for trial in range(trials):
        bandit = NonstatBandit()
        if pol == 'avg':
            agent = Agent(epsilon)
        else:
            agent = AlphaAgent(epsilon, 0.8)
        sum_r = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()  # action : select slot machine
            reward = bandit.play(action)
            agent.update(action, reward)
            sum_r += reward

            rates.append(sum_r / (step+1))

        all_rates[trial] = rates

    avg_rates = np.average(all_rates, axis=0)

    plt.plot(avg_rates, label=pol)
plt.legend()
plt.show()
