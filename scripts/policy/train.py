import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from time import time

from source.sim.launch_app import launch_app
from source.utils.config import load_config

sim_app, args_cli = launch_app(
    enable_cameras=False,
    flatcache=True, 
    mgmt_api=False,
    headless=True,
)

load_config("train")

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.lift import lift_env_cfg

from rsl_rl.runners import OnPolicyRunner

from source.configs.python.environment_cfg import EnvCfg, Env

from isaaclab_rl.rsl_rl.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

import gymnasium as gym

import isaaclab_tasks  # IMPORTANT: registers tasks into gym
from isaaclab_tasks.utils import load_cfg_from_registry

import source.utils.config as config
from source.core.rl.ppo import Actor, Critic
from source.core.rl.rollout import Rollout


policy_cfg = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)

algo_cfg = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.006,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.98,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)

runner_cfg = RslRlOnPolicyRunnerCfg(
    device="cuda:0",
    num_steps_per_env = 24,
    max_iterations = 1500,
    save_interval = 50,
    policy=policy_cfg,
    algorithm=algo_cfg,
)


def main() -> None:
    """
    Main function ran on file execution
    """
    # Create the environment
    env_cfg = EnvCfg()
    env = Env(env_cfg)
    env_wrapper = RslRlVecEnvWrapper(env)
    
    '''env_cfg = load_cfg_from_registry("Isaac-Lift-Cube-Franka-v0", "env_cfg_entry_point")
    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
    env_wrapper = RslRlVecEnvWrapper(env)'''
    
    runner = OnPolicyRunner(
        env=env_wrapper,
        train_cfg=runner_cfg.to_dict(),
        log_dir="source/logs/rsl_rl",
        device=runner_cfg.device,
    )
    
    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
    )
        

if __name__ == "__main__":
    main()