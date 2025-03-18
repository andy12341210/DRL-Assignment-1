import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque



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

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model.eval()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
          return np.random.choice(self.action_size)
        else:
          state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
          q_values = self.model(state_tensor)
          return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device)

        q_values = self.model(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
          max_next_q_values = self.model(next_state_batch).max(1)[0]
          target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=1000)
    env.reset()
    state_size = len(env.get_state())
    action_size = 6
    agent = Agent(state_size, action_size)

    episodes = 5000
    target_update_freq = 10
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.replay()

        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        episode_rewards.append(total_reward)

        if (episode+1) % 100 == 0:
          avg_reward = np.mean(episode_rewards[-100:])
          print(f"Episode {episode+1}/{episodes}, Average Reward: {avg_reward}, Epsilon: {agent.epsilon:.2f}")

    # 儲存模型
    torch.save(agent.model.state_dict(), "taxi_dqn_model.pth")
    print("Model saved successfully.")
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

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model.eval()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
          return np.random.choice(self.action_size)
        else:
          state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
          q_values = self.model(state_tensor)
          return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device)

        q_values = self.model(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
          max_next_q_values = self.model(next_state_batch).max(1)[0]
          target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=1000)
    env.reset()
    state_size = len(env.get_state())
    action_size = 6
    agent = Agent(state_size, action_size)

    episodes = 5000
    target_update_freq = 10
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.replay()

        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        episode_rewards.append(total_reward)

        if (episode+1) % 100 == 0:
          avg_reward = np.mean(episode_rewards[-100:])
          print(f"Episode {episode+1}/{episodes}, Average Reward: {avg_reward}, Epsilon: {agent.epsilon:.2f}")

    # 儲存模型
    torch.save(agent.model.state_dict(), "taxi_dqn_model.pth")
    print("Model saved successfully.")
