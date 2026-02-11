import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from utils.config import config


class Rollout(Dataset):
    """
    Rollout for transition storage
    """
    def __init__(
        self,
        initial_obs: torch.Tensor,
        initial_value_out: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Define collectables
        self.obs: torch.Tensor = initial_obs.unsqueeze(0).to(device=device)
        self.actions: torch.Tensor = torch.empty(0, device=device,)
        self.means: torch.Tensor = torch.empty(0, device=device,)
        self.stds: torch.Tensor = torch.empty(0, device=device,)
        self.rewards: torch.Tensor = torch.empty(0, device=device,)
        self.value_outs: torch.Tensor = initial_value_out.unsqueeze(0).to(device=device)
        self.dones: torch.Tensor = torch.empty(0, device=device,)
        
        
    def __len__(self,) -> int:
        return self.actions.size(0)
    
    
    def __getitem__(
        self,
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return transition tuple
        return (
            self.obs[idx, ...],
            self.actions[idx, ...],
            self.means[idx, ...],
            self.stds[idx, ...],
            self.advantages[idx, ...],
            self.value_outs[idx, ...],
        )
    
    
    def get_horizon(self,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the condensed transition required for advantage computation

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Rewards
                - Dones
                - Value outs
        """
        return  (
            self.rewards,
            self.dones,
            self.value_outs,
        )
        
    
    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        rewards: torch.Tensor,
        value_out: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Add a transition to the rollout

        Args:
            obs (torch.Tensor): Policy observations
            actions (torch.Tensor): Sampled action
            means (torch.Tensor): Policy output means
            stds (torch.Tensor): Policy output stds
            rewards (torch.Tensor): Netted rewards
            value_out (torch.Tensor): Predicted value of the policy
            dones (torch.Tensor): Boolean completion flags
        """
        self.obs = torch.concat((self.obs, obs.unsqueeze(0)), dim=0)
        self.actions = torch.concat((self.actions, actions.unsqueeze(0)), dim=0)
        self.means = torch.concat((self.means, means.unsqueeze(0)), dim=0)
        self.stds = torch.concat((self.stds, stds.unsqueeze(0)), dim=0)
        self.rewards = torch.concat((self.rewards, rewards.unsqueeze(0)), dim=0)
        self.value_outs = torch.concat((self.value_outs, value_out.unsqueeze(0)), dim=0)
        self.dones = torch.concat((self.dones, dones.unsqueeze(0)), dim=0)
        
        
    def add_advantages(
        self,
        advantages: torch.Tensor,
    ) -> None:
        """
        Adds the post-evaluation phase advantages for batching

        Args:
            advantages (torch.Tensor): GAE advantages
        """
        self.advantages = advantages