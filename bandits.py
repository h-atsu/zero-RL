import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        reward = rate > np.random.rand()
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


trials = 2000  # number of experiments trials
steps = 1000
epsilons = [0.5, 0.1, 0.01]
all_rates = np.zeros((trials, steps))


for epsilon in epsilons:
    for trial in range(trials):
        bandit = Bandit()
        agent = Agent(epsilon)
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

    plt.plot(avg_rates)
plt.show()
