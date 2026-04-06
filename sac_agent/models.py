import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """SAC Actor network with CNN for temporal state processing."""
    
    def __init__(self, state_size, action_size, multiplier=1.0):
        super(Actor, self).__init__()
        
        # CNN layers for temporal processing
        self.conv1 = nn.Conv1d(in_channels=state_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Output layers for mean and log_std
        self.fc_mu = nn.Linear(128, action_size)
        self.fc_std = nn.Linear(128, action_size)
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, state_size)
        # Transpose for Conv1d: (batch_size, state_size, time_steps)
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, min=-5, max=0)  # Reduce std range
        std = log_std.exp()
        
        return mu, std


class Critic(nn.Module):
    """SAC Critic network (Q-network) with dual Q-functions."""
    
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        
        # Q1 network
        self.conv1_q1 = nn.Conv1d(in_channels=state_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2_q1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.fc1_q1 = nn.Linear(64 * 5 + action_size, 256)
        self.fc2_q1 = nn.Linear(256, 128)
        self.fc3_q1 = nn.Linear(128, 1)
        
        # Q2 network
        self.conv1_q2 = nn.Conv1d(in_channels=state_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2_q2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.fc1_q2 = nn.Linear(64 * 5 + action_size, 256)
        self.fc2_q2 = nn.Linear(256, 128)
        self.fc3_q2 = nn.Linear(128, 1)
        
    def forward(self, state, action):
        # Q1 forward
        x1 = state.transpose(1, 2)
        x1 = F.relu(self.conv1_q1(x1))
        x1 = F.relu(self.conv2_q1(x1))
        x1 = x1.reshape(x1.size(0), -1)
        x1 = torch.cat([x1, action], dim=1)
        x1 = F.relu(self.fc1_q1(x1))
        x1 = F.relu(self.fc2_q1(x1))
        q1 = self.fc3_q1(x1)
        
        # Q2 forward
        x2 = state.transpose(1, 2)
        x2 = F.relu(self.conv1_q2(x2))
        x2 = F.relu(self.conv2_q2(x2))
        x2 = x2.reshape(x2.size(0), -1)
        x2 = torch.cat([x2, action], dim=1)
        x2 = F.relu(self.fc1_q2(x2))
        x2 = F.relu(self.fc2_q2(x2))
        q2 = self.fc3_q2(x2)
        
        return q1, q2
