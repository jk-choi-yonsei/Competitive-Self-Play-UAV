from typing import Tuple
import numpy as np
import torch


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores (state, action, log_prob, reward, done, value) for one episode or N steps.
    """

    def __init__(self, capacity: int, state_shape: Tuple, action_size: int, device: torch.device):
        """
        capacity: max number of steps to store
        state_shape: (time_steps, state_size) - shape of a single stacked observation
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.ptr = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.ptr += 1

    def __len__(self):
        return self.ptr

    def compute_returns_and_advantages(self, last_value: float, gamma: float, lam: float):
        """
        GAE (Generalized Advantage Estimation) computation.
        last_value: V(s_T) for bootstrap (0 if terminal).
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t + 1] if t + 1 < len(dones) else 1.0)
            delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def get_tensors(self, returns: np.ndarray, advantages: np.ndarray):
        """Return all stored data as tensors for PPO update."""
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        return states_t, actions_t, log_probs_t, returns_t, advantages_t
