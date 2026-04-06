import math
import numpy as np
import torch


def get_action(mu, std, multiplier=1.0):
    """Sample action from the policy distribution (GPU-efficient version)."""
    # Sample on GPU
    normal = torch.distributions.Normal(mu, std)
    z = normal.rsample()
    action_raw = torch.tanh(z)  # [-1, 1]

    # Throttle (last dim) allowed to be [-1, 1] for Speed Brake control
    # Previously it was 0.5 * (tanh + 1) -> [0, 1]
    # Now we treat it same as other controls if we want full range [-1, 1]
    
    # action = action_raw.clone() # No longer needed if all are [-1, 1]
    
    return action_raw.detach().cpu().numpy()


def eval_action(mu, std, multiplier=1.0):
    """Evaluate action with reparameterization trick and correct log_prob calculation."""
    normal = torch.distributions.Normal(mu, std)
    z = normal.rsample()
    action_raw = torch.tanh(z)  # [-1, 1]

    # Throttle allowed [-1, 1] for Speed Brake
    action = action_raw # .clone() not strictly needed

    # Compute log probability with squashing
    # All dims are just tanh(z), so squash correction is same for all
    log_prob = normal.log_prob(z)
    squash = torch.log(1 - action_raw.pow(2) + 1e-6)
    
    log_prob = log_prob - squash
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob


def hard_target_update(source, target):
    """Hard update: copy all parameters from source to target."""
    target.load_state_dict(source.state_dict())


def soft_target_update(source, target, tau=0.005):
    """Soft update: slowly blend source parameters into target."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
