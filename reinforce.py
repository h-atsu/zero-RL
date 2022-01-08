from collections import deque, defaultdict
import random
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from typing import Collection
from gridworld import GridWorld
from mcpolicy import greedy_action_probs
import matplotlib.pyplot as plt
import copy
import time
import imageio


class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            # prob は方策の実現値として行動が得られる確率
            G = reward + self.gamma*G
            loss += -F.log(prob) * G
        loss.backward()
        self.optimizer.update()
        self.memory = []


episodes = 3000
env = gym.make('CartPole-v0')
agent = Agent()
reward_log = []

for episode in range(episodes):
    state = env.reset()
    done = False
    sum_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.add(reward, prob)
        state = next_state
        sum_reward += reward

    agent.update()

    reward_log.append(sum_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, sum_reward))


plt.plot(reward_log)
plt.show()

# 可視化パート
state = env.reset()
done = False
sum_reward = 0
images = []
while not done:
    time.sleep(0.01)
    action, prob = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    sum_reward += reward
    env.render()
    images.append(env.render(mode='rgb_array'))
print("Total Rewards : ", sum_reward)
imageio.mimsave('cartpole_reinforce.gif', images,
                'GIF', **{'duration': 1.0/30.})
env.close()
