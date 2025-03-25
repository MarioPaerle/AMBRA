import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch import optim
from environment import Game
import random as rd
import time
import matplotlib.pyplot as plt

"""
Basic Copied Implementation of Deep Q Network to Study.
Pretty Simple tho

"""



class LinearBrain1(nn.Module):
    def __init__(self, n, o):
        super(LinearBrain1, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(n*n, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, o)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.n**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class ConvBrain1(nn.Module):
    def __init__(self, n, o):
        super(ConvBrain1, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(1, 256, 4, padding='same')
        self.conv2 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv3 = nn.Conv2d(256, 32, 2, padding='same')

        self.l1 = nn.Linear(n*n*32, 384)
        self.l2 = nn.Linear(384, o)

    def forward(self, x):
        x = x.view(x.shape[0], 1, self.n, self.n)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x

class SimpleRLAgent:
    def __init__(self, state_dim, action_dim, brain):
        self.state_dim = state_dim
        self.action_dim = action_dim

        BRAIN = brain
        self.q_network = BRAIN(state_dim, action_dim)
        self.target_network = BRAIN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.002)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

        # EXPLORATION RATE
        self.epsilon = 1.0
        self.epsilon_min = 0.12
        self.epsilon_decay = 0.995


        # LOSS
        self.rewards = []

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q  # 0.99 is discount factor

        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        # Soft update (you could also do hard updates periodically)
        tau = 0.01
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))


    def train(self, enemy, env, episodes=400):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            total_enemy_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward1, reward2, done = env.step(action)

                enemy_action = enemy.get_action(next_state)
                enemy_next_state, enemy_reward1, enemy_reward2, enemy_done = env.step(enemy_action + 12)

                self.store_experience(state, action, reward1, next_state, done)
                enemy.store_experience(next_state, enemy_action, enemy_reward1, enemy_next_state, enemy_done)

                self.update_network()
                enemy.update_network()

                state = next_state
                total_reward += reward1 # + enemy_reward2
                total_enemy_reward += enemy_reward1 # + reward2

            self.rewards.append(total_reward)
            enemy.rewards.append(total_enemy_reward)

            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

class _SimpleRLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        BRAIN = ConvBrain1
        BRAIN2 = ConvBrain1

        # Initialize Q-network1
        self.q_network = BRAIN(state_dim, action_dim)
        self.target_network = BRAIN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.002)
        self.loss_fn = nn.MSELoss()

        # Initialize Q-network2
        self.q_network2 = BRAIN2(state_dim, action_dim)
        self.target_network2 = BRAIN2(state_dim, action_dim)
        self.target_network2.load_state_dict(self.q_network2.state_dict())
        self.optimizer2 = optim.Adam(self.q_network2.parameters(), lr=0.002)
        self.loss_fn2 = nn.MSELoss()
        self.replay_buffer2 = deque(maxlen=10000)
        self.batch_size2 = 64
        self.epsilon2 = 1.0
        self.epsilon_min2 = 0.05
        self.epsilon_decay2 = 0.9975

        # REPLAY BUFFER SIZE
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

        # EXPLORATION RATE
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9975


        # LOSS
        self.rewards = []
        self.rewards2 = []

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def get_action2(self, state):
        if random.random() < self.epsilon2:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network2(state)
        return torch.argmax(q_values).item()

    def update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q  # 0.99 is discount factor

        # Compute loss and update
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_network2(self):
        if len(self.replay_buffer2) < self.batch_size2:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer2, self.batch_size2)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute Q values
        current_q = self.q_network2(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network2(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q  # 0.99 is discount factor

        # Compute loss and update
        loss = self.loss_fn2(current_q.squeeze(), target_q)
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer2.step()

        # Update target network
        self.update_target_network2()

        # Decay epsilon
        self.epsilon2 = max(self.epsilon_min2, self.epsilon2 * self.epsilon_decay2)

    def update_target_network(self):
        # Soft update (you could also do hard updates periodically)
        tau = 0.01
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_target_network2(self):
        # Soft update (you could also do hard updates periodically)
        tau = 0.01
        for target_param, param in zip(self.target_network2.parameters(),
                                       self.q_network2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def store_experience2(self, state, action, reward, next_state, done):
        self.replay_buffer2.append((state, action, reward, next_state, done))

    def to_other_player(self, state2):
        state2 = -state2
        state2 = np.flip(state2).copy()
        return state2


    # Usage with your custom environment
    def train(self, env, episodes=500):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            total_reward2 = 0
            done = False
            moves = 0
            while not done:
                moves += 1
                # Get action from agent
                action = self.get_action(state)

                # Take action in environment
                next_state, reward, reward2, done1 = env.step(action, opponent=False)
                action2 = self.get_action2(self.to_other_player(next_state)) + 12
                next_state, reward02, reward1, done2 = env.step(action2, opponent=True)
                reward += reward1
                reward2 +=reward02
                # Store experience
                self.store_experience(state, action, reward, next_state, done)

                # Update network
                self.update_network()
                total_reward += reward


                self.store_experience2(state, action, reward2, next_state, done)
                self.update_network2()
                state = next_state
                total_reward2 += reward2
                done = done1 or done2

            self.rewards.append(total_reward)
            self.rewards2.append(total_reward2)

            print(f"Episode {episode}, Total Reward 1: {total_reward}, Total Reward 2: {total_reward2} Epsilon: {agent.epsilon:.2f}, Moves: {moves}")

def moving_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

env = Game(dimension=5, render=True)
state_dim = 5
action_dim = 12
agent = SimpleRLAgent(state_dim, action_dim, brain=ConvBrain1)
agent2 = SimpleRLAgent(state_dim, action_dim, brain=LinearBrain1)

agent.train(agent2, env)
plt.plot(moving_average(agent.rewards, 10))
plt.plot(moving_average(agent2.rewards, 10))
plt.show()