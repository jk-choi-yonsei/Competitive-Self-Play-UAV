import numpy as np
import random
import torch


class ReplayBuffer:
    """Efficient replay buffer for SAC training."""
    
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """Sample batch of experiences and return as GPU tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        masks = torch.from_numpy(1.0 - self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, masks
        
    def __len__(self):
        return self.size
