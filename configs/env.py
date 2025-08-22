import math
import torch
import yaml

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class EnvCfg(DirectRLEnvCfg):
    '''
    Environment configuration
    '''
    # Read YAML file
    with open('hyperparams.yaml', 'r') as f:
        params = yaml.safe_load(f)
        
    # Env config
    decimation: int = params['env']['decimation']
    episode_length: int = params['env']['episode_length'] # [sec]
    action_scale = params['env']['action_scale']
    angle_obs_space = params['env']['angle_obs_space']
    vel_obs_space = params['env']['velocity_obs_space']
    state_space = params['env']['state_space']
    
    # Sim config
    sim: SimulationCfg = SimulationCfg(dt=1/100)
    
    # Panda config
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path='/Word/envs/env_.*/Panda')
    robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
    robot_cfg.actuators["panda_shoulder"].damping = 0.0
    robot_cfg.actuators["panda_forearm"].stiffness = 0.0
    robot_cfg.actuators["panda_forearm"].damping = 0.0