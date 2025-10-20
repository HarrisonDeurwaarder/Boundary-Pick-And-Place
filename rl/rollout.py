import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


class Rollout(Dataset):
    '''
    Stores transition tuples for policy/value training
    '''
    def __init__(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        rewards: list[float],
        distributions: list[Normal],
    ) -> None:
        '''
        Args:
            states (list[Tensor]): Panda joint orientation/velocity, depth-processed images
            actions (list[Tensor]): Sampled end-effector position
            rewards (list[float]): Timestep rewards
            distributions (list[Normal]): Distributions output by the policy; actions are sampled here
        '''
        self.states = states[:-1]
        self.next_states = states[1:]
        self.actions = actions
        self.rewards = rewards
        self.distributions = distributions
        
    
    def __len__(self,) -> int:
        return len(self.rewards)
    
    
    def __getitem__(
        self,
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Normal]:
        return (
            self.states[idx],
            self.next_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.distributions[idx],
        )