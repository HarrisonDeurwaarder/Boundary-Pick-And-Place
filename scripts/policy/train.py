import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from time import time

from source.sim.launch_app import launch_app
from source.utils.config import load_config

sim_app, args_cli = launch_app(
    enable_cameras=True,
    flatcache=True, 
    mgmt_api=True,
    #headless=True,
)
load_config("train")

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation

from source.core.rl.environment import Env

import source.utils.config as config
from source.core.rl.ppo import Actor, Critic
from source.core.rl.rollout import Rollout


def main() -> None:
    """
    Main function ran on file execution
    """
    # Set the correct seed given the argument
    torch.seed()
    # Create the environment
    env: Env = Env()
    # Load the simulation
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(env.cfg.sim,)
    sim.reset()
    # Extract the scene and robot
    scene: InteractiveScene = env.scene
    device: torch.device = env.cfg.sim.device
    
    # Create RL objects
    policy: Actor = Actor().to(device=device)
    value: Critic = Critic().to(device=device)
    
    optimizer: Adam = Adam([
            {"params": policy.parameters(), "lr": config.config["rl"]["ppo"]["policy_lr"]},
            {"params": value.parameters(), "lr": config.config["rl"]["ppo"]["value_lr"]},
        ],
    )
    
    """ Training Loop """
    # Initial reset
    obs: torch.Tensor = env.reset()[0]
    rollout: Rollout = Rollout(
        initial_obs=obs,
        initial_value_out=value(obs,),
        device=device,
    )
    # Epoch = num rollouts collected
    for iteration in range(config.config["rl"]["iterations"]):
        start = time()
        # Rollout collection phase
        rollout.reset()
        # Loop until rollout is at capacity
        while len(rollout) < config.config["rl"]["rollout_length"]:
            # Sample action from policy
            mean, std = policy(obs,)
            action: torch.Tensor = Normal(mean, std,).sample()
            # Step in the environment
            obs, rew, term, trunc, _, = env.step(action,)
            # Add to rollout
            rollout.add(
                obs, action, mean, std, rew, value(obs,), term | trunc,
            )
            # Update the scene and rendering
            scene.update(env.physics_dt,)
            sim_app.update()
        
        # Batch rollout
        dataloader: DataLoader = DataLoader(
            dataset=rollout,
            batch_size=config.config["rl"]["batch_size"],
            shuffle=True,
        )
        
        # Training phase
        rews_h, dones_h, value_outs_h = rollout.get_horizon()
        # Compute advantages
        advantages: torch.Tensor = policy.gae(
            rewards=rews_h,
            dones=dones_h,
            value_outs=value_outs_h,
        )
        rollout.add_advantages(advantages,)
        
        for _ in range(config.config["rl"]["epochs"]):
            for _obs, actions, old_means, old_stds, advantages, old_value_outs in dataloader:
                optimizer.zero_grad()
                # Compute advantages and loss
                value_outs: torch.Tensor = value(_obs,).squeeze(2)
                means, stds = policy(_obs,)
                policy_dist: Normal = Normal(means, stds)
                
                policy_objective: torch.Tensor = policy.policy_objective(
                    policy_dist=policy_dist,
                    old_policy_dist=Normal(old_means, old_stds),
                    actions=actions,
                    advantages=advantages,
                )
                value_loss: torch.Tensor = value.value_loss(
                    value_outs=value_outs,
                    old_value_outs=old_value_outs,
                    advantages=advantages,
                )
                
                # Backpropagate
                loss: torch.Tensor = -policy_objective + config.config["rl"]["ppo"]["value_coef"] * value_loss - config.config["rl"]["ppo"]["entropy_coef"] * policy_dist.entropy().mean()
                loss.backward()
                optimizer.step()
                # Continue scene interaction
                scene.update(env.physics_dt,)
                sim_app.update()
                
        print(
            "\n========================================\n",
            f"Iteration #{iteration+1}/{config.config['rl']['iterations']} Completed:",
            f"\tMean Reward: {rollout.rewards.mean():.2f}",
            f"\tMean Episode Length: {((1 - rollout.dones).sum()) / rollout.dones.sum():.2f}",
            f"\tMean Advantage: {rollout.advantages.mean():.2f}",
            f"\tSTD Advantage: {rollout.advantages.std():.2f}",
            f"\tMean Value Loss: {Critic.value_loss(value(rollout.obs[:-1],).squeeze(2), rollout.value_outs[:-1], rollout.advantages,).mean():.2f}",
            f"\tMean Policy Objective: {Actor.policy_objective(Normal(*policy(rollout.obs[:-1])), Normal(rollout.means, rollout.stds), rollout.actions, rollout.advantages).mean():.2f}",
            *(f"\tMean Rewards/{key}: {(rewards.mean() / config.config['rl']['rollout_length']):.2f}" for key, rewards in env._episode_rewards.items()),
            sep="\n",
        )
        print(f'\nETA: {((time() - start) * (config.config["rl"]["iterations"] - iteration - 1) / (3600.0)):.2f} hours')
        

if __name__ == "__main__":
    main()