import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActor(nn.Module):
    """PPO Actor network with CNN for temporal state processing (same architecture as SAC Actor)."""

    def __init__(self, state_size: int, action_size: int):
        super(PPOActor, self).__init__()

        # CNN layers for temporal processing (stack_size=5 time steps)
        self.conv1 = nn.Conv1d(in_channels=state_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output: mean and log_std for Gaussian policy
        self.fc_mu = nn.Linear(128, action_size)
        # log_std as learnable parameter (shared across all states, common PPO practice)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        """
        x: (batch, time_steps, state_size)
        Returns: mu (batch, action_size), std (batch, action_size)
        """
        # Transpose for Conv1d: (batch, state_size, time_steps)
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.fc_mu(x)
        # Clamp log_std for stability
        log_std = torch.clamp(self.log_std, min=-3, max=0)
        std = log_std.exp().expand_as(mu)

        return mu, std


class PPOCritic(nn.Module):
    """PPO Critic network (value function) with CNN for temporal state processing."""

    def __init__(self, state_size: int):
        super(PPOCritic, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=state_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_val = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: (batch, time_steps, state_size)
        Returns: value (batch, 1)
        """
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.fc_val(x)
        return value
