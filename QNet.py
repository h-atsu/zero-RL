import numpy as np
import torch
from torch._C import dtype
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Collection
from gridworld import GridWorld
from mcpolicy import greedy_action_probs
import matplotlib.pyplot as plt


def one_hot(state):
    """
    state : (y,x) のみに1それ以外に0の one-hot-encoding
    """
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return torch.tensor(vec[np.newaxis, :])


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.l1 = nn.Linear(12, 100)
        self.l2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x.double()


class QLerningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return torch.argmax(qs).item()

    def update(self, state_vec, action, reward, next_state_vec, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state_vec).detach().numpy()
            next_q = np.max(next_qs, axis=1)

        target = torch.tensor(self.gamma * next_q + reward)
        qs = self.qnet(state_vec)
        q = qs[:, action]
        loss = nn.MSELoss()(target, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


env = GridWorld()
agent = QLerningAgent()


episodes = 1000
loss_list = []

for episode in range(episodes):
    state = env.reset()
    state_vec = one_hot(state)
    avg_loss = []

    while(True):
        action = agent.get_action(state_vec)
        next_state, reward, done = env.step(action)

        next_state_vec = one_hot(next_state)
        loss = agent.update(state_vec, action, reward, next_state_vec, done)
        avg_loss.append(loss)
        if done:
            loss_list.append(np.average(avg_loss))

            break
        state_vec = next_state_vec

    # print(loss)

plt.plot(range(len(loss_list)), loss_list)
plt.show()
