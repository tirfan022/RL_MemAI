import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from agent.network import CacheQNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CacheQNetwork(state_dim, action_dim).to(self.device)
        self.target_model = CacheQNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=20000)
        self.batch_size = 64

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values, dim=1).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        with torch.no_grad():
            next_state_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, next_state_actions).squeeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
            
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)