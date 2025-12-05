import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_config


CONFIG = load_config('panda_train')


class Actor(nn.Module):
    '''
    The policy network
    '''
    def __init__(self,) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
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
    ) -> torch.distributions.Normal:
        '''
        Pass a state through the policy for an action
        
        Args:
            state (Tensor): Location/velocity data of the panda along with boundary vector
            
        Returns:
            Tensor: OSC-required inputs
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
            Tensor: Low variance/low bias advantage estimations
        '''
        # Pre-compute the TD residuals
        td_residuals: torch.Tensor = rewards + critic_out[1:] * CONFIG['rl']['ppo']['discount_factor'] - critic_out[:-1]
        # Iteratively compute GAE
        advantages: torch.Tensor = torch.zeros_like(rewards)
        for t in reversed(td_residuals.size(-1) - 1):
            advantages[t] = td_residuals[..., t] + CONFIG['rl']['ppo']['discount_factor'] * CONFIG['rl']['ppo']['gae_decay'] * (1 - dones[..., t+1]) * advantages[..., t+1]
        
        return advantages
        
    
    @classmethod
    def policy_objective(
        cls,
        policy_dist: torch.distributions.Normal,
        old_policy_dist: torch.distributions.Normal,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        clipping_param: float = 0.2,
    ) -> torch.Tensor:
        '''
        Computes the PPO policy function objective
        
        Args:
            policy_dist (Normal): Current policy's output distribution given the state
            old_policy_dist (Normal): Target policy's output distribution given the state
            actions (Tensor): Continuous action space sampled by the frozen policy and conducted by the agent
            advantages (Tensor): GAE advantages
            clipping_param (float): Bounds in which clip the policy ratio
        
        Returns:
            Tensor: PPO policy objected derived from advantage estimations with entropy bonus
        '''
        # Ratio of probabilities of the selected action (mean log probs for multiple continous actions)
        policy_ratio: torch.Tensor = torch.exp(policy_dist.log_prob(actions) - old_policy_dist.log_prob(actions))
        # Compute underestimate for policy performance
        clipped_surrogate_obj: torch.Tensor = torch.minimum(
            advantages * policy_ratio,
            advantages * torch.clip(
                policy_ratio,
                1 - clipping_param,
                1 + clipping_param
            ),
        )
        # Compute entropy term for added exploration
        entropy: torch.Tensor = policy_dist.entropy()
        
        return -torch.mean(
            clipped_surrogate_obj + CONFIG['rl']['ppo']['entropy_coefficient'] * entropy,
            dim=-1,
        )
    
    
class Critic(nn.Module):
    '''
    The value network
    '''
    def __init__(self,) -> None:
        super().__init__()
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
            Tensor: OSC-required inputs
        '''
        return self.net(state,)
    
    
    @classmethod
    def critic_objective(
        cls,
        critic_outs: torch.Tensor,
        old_critic_outs: torch.Tensor,
        advanatges: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute the PPO value function objective
        
        Args:
            critic_outs (Tensor): Predicted value of the current policy
            old_critic_outs (Tensor): Predicted value of the target policy
            advantages (Tensor): GAE advantages
            
        Returns:
            Tensor: MSE critic objective
        '''
        target_value: torch.Tensor = advanatges + old_critic_outs
        return F.mse_loss(
            critic_outs,
            target_value,
        )