import yaml

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg


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
    episode_length: float = params['env']['episode_length'] # [sec]
    action_scale: float = params['env']['action_scale']
    angle_obs_space: int = params['env']['angle_obs_space']
    vel_obs_space: int = params['env']['velocity_obs_space']
    state_space: int = params['env']['state_space']
    
    # Sim config
    sim: SimulationCfg = SimulationCfg(dt=1/100)
    
    # Fill in later
    joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]
    
    # Scene config
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=params['scene']['num_envs'],
        env_spacing=params['scene']['env_spacing'],
        replicate_physics=params['scene']['replicate_physics'],
        clone_in_fabric=params['scene']['clone_in_fabric'],
    )
    
    # Reset config
    joint_range: list[int] = params['reset']['joint_range']
    displ_range: list[int] = params['reset']['object_displacement_range']
    
    # Reward scale config
    rew_scale_grasp: float = params['reward']['scale_grasp']
    rew_scale_duration: float = params['reward']['scale_duration']
    rew_scale_distance: float = params['reward']['scale_distance']
    rew_scale_drop: float = params['reward']['scale_drop']
    rew_scale_contact: float = params['reward']['scale_contact']