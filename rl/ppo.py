import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    '''
    The policy network
    '''
    def __init__(self,) -> None:
        self.net: nn.Sequential = nn.Sequential(
            ...
        )
    
    
    def __call__(self,
                 state: torch.Tensor,) -> torch.Tensor:
        return super().__call__(state)
    
    
    def forward(self,
                state: torch.Tensor,) -> torch.distributions.Normal:
        '''
        Pass a state through the policy for an action
        
        Args:
            state (Tensor): Location/velocity data of the panda along with boundary vector
            
        Returns:
            action (Tensor): OSC-required inputs
        '''
        # Chunk the outputs into distribution parameters
        dist_params = torch.chunk(self.net(state), chunks=2, dim=-1)
        # Return distributions
        return torch.distributions.Normal(
            loc=dist_params[..., 0], 
            scale=torch.exp(dist_params[..., 1]),
        )
    
    
    @classmethod
    def gae(
        cls,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        critic_out: torch.Tensor,
        discount_factor: float = 0.99,
        gae_decay: float = 0.98,
    ) -> torch.Tensor:
        '''
        Computes GAE, a low variance/low bias advantage estimation function
        
        Args:
            rewards (Tensor): Float environment rewards at every step in the rollout
            dones (Tensor): Boolean flags indicating if an episode has been terminated at every rollout
            critic_out (Tensor): Predicted value derived from the critic
            discount_factor (float): MDP discount factor to weight more recent rewards higher in the value computation
            gae_decay (float): GAE hyperparameter to control TD propagation
        
        Returns:
            advantage (Tensor): Low variance/low bias advantage estimations
        '''
        # Pre-compute the TD residuals
        td_residuals: torch.Tensor = rewards + critic_out[1:] * discount_factor - critic_out[:-1]
        # Iteratively compute GAE
        advantages: torch.Tensor = torch.zeros_like(rewards)
        for t in reversed(td_residuals.size(-1) - 1):
            advantages[t] = td_residuals[..., t] + discount_factor * gae_decay * (1 - dones[..., t+1]) * advantages[..., t+1]
        
        return advantages
        
    
    @classmethod
    def surrogate_obj(
        cls,
        policy_dist: torch.distributions.Normal,
        old_policy_dist: torch.distributions.Normal,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        critic_out: torch.Tensor,
        discount_factor: float = 0.99,
        gae_decay: float = 0.98,
        clipping_param: float = 0.2,
    ) -> torch.Tensor:
        '''
        Computes the PPO objective as a static method
        Also computes the value target for the critic
        
        Args:
            policy_dist (Normal): Current policy's output distribution given the state
            old_policy_dist (Normal): Frozen policy's output distribution given the state
            actions (Tensor): Continuous action space sampled by the frozen policy and conducted by the agent
            rewards (Tensor): Float environment rewards at every step in the rollout
            dones (Tensor): Boolean flags indicating if an episode has been terminated at every rollout
            critic_out (Tensor): Predicted value derived from the critic
            discount_factor (float): MDP discount factor to weight more recent rewards higher in the value computation
            gae_decay (float): GAE hyperparameter to control TD propagation
            clipping_param (float): Bounds in which clip the policy ratio
        
        Returns:
            clipped_surrogate_obj (Tensor): PPO policy objected derived from advantage estimations
            advantage (Tensor): Low variance/low bias advantage estimations
        '''
        # Ratio of probabilities of the selected action (mean log probs for multiple continous actions)
        policy_ratio: torch.Tensor = torch.exp(policy_dist.log_prob(actions) - old_policy_dist.log_prob(actions))
        # Advantage and value target computation
        advantages = Actor.gae(
            rewards=rewards,
            dones=dones,
            predicted_value=critic_out,
            discount_factor=discount_factor,
            gae_decay=gae_decay,
        )
        # Scaling the advantages
        clipped_surrogate_obj: torch.Tensor = torch.minimum(
            advantages * policy_ratio,
            advantages * torch.clip(
                policy_ratio,
                1 - clipping_param,
                1 + clipping_param
            ),
        )
        
        return clipped_surrogate_obj, advantages
    
    
class Critic(nn.Module):
    '''
    The policy network
    '''
    def __init__(self,) -> None:
        self.net = nn.Sequential(
            ...
        )
    
    
    def __call__(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return super().__call__(state)
    
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Pass a state through the policy for an action
        
        Args:
            state (Tensor): Location/velocity data of the panda along with boundary vector
            
        Returns:
            action (Tensor): OSC-required inputs
        '''
        return self.net(state,)