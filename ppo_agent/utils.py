import numpy as np
import torch


def get_action_ppo(mu: torch.Tensor, std: torch.Tensor):
    """
    Sample action from Gaussian policy and compute its log probability.
    Uses tanh squashing: action = tanh(z), z ~ N(mu, std).
    Returns:
        action_np: np.ndarray with shape (action_size,)
        log_prob: scalar float
    """
    normal = torch.distributions.Normal(mu, std)
    z = normal.rsample()
    action_raw = torch.tanh(z)

    # Log prob with squashing correction (same as SAC)
    log_prob = normal.log_prob(z)
    squash = torch.log(1.0 - action_raw.pow(2) + 1e-6)
    log_prob = (log_prob - squash).sum(dim=-1)  # scalar

    action_np = action_raw.detach().cpu().numpy()
    log_prob_val = log_prob.detach().cpu().item()
    return action_np, log_prob_val


def eval_action_ppo(mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
    """
    Re-evaluate log_prob and entropy for given actions (used in PPO update).
    actions: previously sampled (pre-tanh) can't be stored; we store tanh(z) actions.
    So we treat stored actions as tanh(z) and recover z = atanh(action).
    Returns:
        log_probs: (batch,) log probabilities
        entropy: (batch,) approximate entropy
    """
    # Clamp to avoid atanh divergence at +-1
    actions_clamped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    z = torch.atanh(actions_clamped)

    normal = torch.distributions.Normal(mu, std)
    log_prob = normal.log_prob(z)
    squash = torch.log(1.0 - actions_clamped.pow(2) + 1e-6)
    log_prob = (log_prob - squash).sum(dim=-1)  # (batch,)

    # Approximate entropy (via differential entropy of Gaussian - squash correction)
    entropy = normal.entropy().sum(dim=-1)  # (batch,)
    return log_prob, entropy
