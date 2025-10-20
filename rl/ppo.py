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
        self,
    ) -> torch.Tensor:
        '''
        Computes GAE
        '''
        pass
    
    
    @classmethod
    def surrogate_obj(
        self,
        policy_dist: torch.distributions.Normal,
        old_policy_dist: torch.distributions.Normal,
        actions: torch.Tensor,
        critic_out: torch.Tensor,
        clipping_param: float,
    ) -> torch.Tensor:
        '''
        Computes the PPO objective as a static method
        
        Args:
            policy_dist (Normal): Current policy's output distribution given the state
            old_policy_dist (Normal): Frozen policy's output distribution given the state
            actions (Tensor): Continuous action space sampled by the frozen policy and conducted by the agent
            critic_out (Tensor): Predicted value of the policy's actions
        '''
        # Disable gradient computations
        # Gradients shouldn't flow through policy ratio or advantage computation
        with torch.no_grad():
            # Ratio of probabilities of the selected action (mean log probs for multiple continous actions)
            policy_ratio: torch.Tensor = torch.exp(policy_dist.log_prob(actions) - old_policy_dist.log_prob(actions))
            gae: torch.Tensor = Actor.gae()
            
        clipped_surrogate_obj: torch.Tensor = torch.minimum(
            gae * policy_ratio,
            gae * torch.clip(
                policy_ratio,
                1 - clipping_param,
                1 + clipping_param
            ),
        )
    
    
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