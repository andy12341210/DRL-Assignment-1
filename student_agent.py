# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import dill
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
# from training import DQN, Agent

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # 折扣係數
        self.epsilon = 1.0  # 初始探索機率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.memory = deque(maxlen=2000)

        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model.eval()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
          return np.random.choice(self.action_size)
        else:
          state_tensor = torch.FloatTensor(state).unsqueeze(0)
          q_values = self.model(state_tensor)
          return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        q_values = self.model(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
          max_next_q_values = self.model(next_state_batch).max(1)[0]
          target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

Q_table = None

with open("taxi_dqn_model.pkl", "rb") as f:
    Q_table = dill.load(f, ignore=True)

print(Q_table)

def get_action(obs):
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        q_values = Q_table(state_tensor)
    return torch.argmax(q_values).item()

    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

