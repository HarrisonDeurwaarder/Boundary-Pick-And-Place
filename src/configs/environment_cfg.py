import yaml

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from src.utils.hyperparams import HPARAMS
from src.configs.scene_cfg import SceneCfg
from src.configs.event_cfg import EventCfg


@configclass
class EnvCfg(DirectRLEnvCfg):
    '''
    Environment configuration
    '''
    # Env config
    decimation: int = HPARAMS['env']['decimation']
    episode_length: float = HPARAMS['env']['episode_length'] # [sec]
    action_scale: float = HPARAMS['env']['action_scale']
    angle_obs_space: int = HPARAMS['env']['angle_obs_space']
    vel_obs_space: int = HPARAMS['env']['velocity_obs_space']
    state_space: int = HPARAMS['env']['state_space']
    
    # Sim config
    sim: SimulationCfg = SimulationCfg(dt=1/100)
    
    # Event config
    events: EventCfg = EventCfg()
    
    # Scene config
    scene: SceneCfg = SceneCfg(
        num_envs=HPARAMS['scene']['num_envs'],
        env_spacing=HPARAMS['scene']['env_spacing'],
        replicate_physics=HPARAMS['scene']['replicate_physics'],
        clone_in_fabric=HPARAMS['scene']['clone_in_fabric'],
    )
    
    # Reset config
    joint_range: list[int] = HPARAMS['reset']['joint_range']
    displ_range: list[int] = HPARAMS['reset']['object_displacement_range']
    
    # Reward scale config
    rew_scale_grasp: float = HPARAMS['reward']['scale_grasp']
    rew_scale_duration: float = HPARAMS['reward']['scale_duration']
    rew_scale_distance: float = HPARAMS['reward']['scale_distance']
    rew_scale_drop: float = HPARAMS['reward']['scale_drop']
    rew_scale_contact: float = HPARAMS['reward']['scale_contact']