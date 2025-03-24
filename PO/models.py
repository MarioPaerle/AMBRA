import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch import optim
from environment import Game
import random as rd
import matplotlib.pyplot as plt

"""
Basic Copied Implementation of Deep Q Network to Study.
Pretty Simple tho

"""



class LinearBrain1(nn.Module):
    def __init__(self, n, o):
        super(LinearBrain1, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(n*n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, o)

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
        self.conv2 = nn.Conv2d(256, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 32, 3, padding='same')

        self.l1 = nn.Linear(n*n*32, 256)
        self.l2 = nn.Linear(256, o)

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
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        BRAIN = ConvBrain1

        # Initialize Q-network
        self.q_network = BRAIN(state_dim, action_dim)
        self.target_network = BRAIN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.002)
        self.loss_fn = nn.MSELoss()

        # REPLAY BUFFER SIZE
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

    def update_target_network(self):
        # Soft update (you could also do hard updates periodically)
        tau = 0.01
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))


    # Usage with your custom environment
    def train(self, env, episodes=100):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Get action from agent
                action = self.get_action(state)

                # Take action in environment
                next_state, reward, done = env.step(action)
                env.step(rd.randint(12, 23))

                # Store experience
                self.store_experience(state, action, reward, next_state, done)

                # Update network
                self.update_network()

                state = next_state
                total_reward += reward

            self.rewards.append(total_reward)

            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

env = Game(dimension=5, render=False)
state_dim = 5
action_dim = 12
agent = SimpleRLAgent(state_dim, action_dim)
agent.train(env)

plt.plot(agent.rewards)
plt.show()