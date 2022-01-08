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


class ReplayBuffer:
    def __init__(self, buffuer_size=10000, batch_size=32):
        self.buffer = deque(maxlen=buffuer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.stack([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.05
        self.buffer_size = 1000000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        qs = self.qnet(state)
        qs_next = self.qnet(next_state)
        max_action = np.argmax(qs_next.data, axis=1)

        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs[np.arange(self.batch_size), max_action]
        next_q.unchain()
        td_target = reward + (1-done) * self.gamma * next_q

        loss = F.mean_squared_error(q, td_target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


episodes = 200
sync_interval = 20
env = gym.make('CartPole-v0')
agent = DQNAgent()
reward_log = []

for episode in range(episodes):
    state = env.reset()
    done = False
    sum_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        sum_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_log.append(sum_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}, epsilon: {}".format(
            episode, sum_reward, agent.epsilon))


plt.plot(reward_log)
plt.show()

# 可視化パート
agent.epsilon = 0
state = env.reset()
done = False
sum_reward = 0
images = []
while not done:
    time.sleep(0.01)
    action = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    sum_reward += reward
    env.render()
    images.append(env.render(mode='rgb_array'))
print("Total Rewards : ", sum_reward)
imageio.mimsave('cartpole.gif', images, 'GIF', **{'duration': 1.0/30.})
env.close()
