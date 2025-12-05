from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from utils.config import load_config
from configs.python.scene_cfg import SceneCfg
from configs.python.event_cfg import EventCfg


CONFIG = load_config('panda_train')


@configclass
class EnvCfg(DirectRLEnvCfg):
    '''
    Environment configuration
    '''
    # Env config
    decimation: int = CONFIG['env']['decimation']
    episode_length_s: float = CONFIG['env']['episode_length'] # [sec]
    action_scale: float = CONFIG['env']['action_scale']
    action_space: float = CONFIG['env']['action_space']
    observation_space: int = CONFIG['env']['angle_obs_space'] + CONFIG['env']['velocity_obs_space']
    state_space: int = CONFIG['env']['state_space']
    
    # Sim config
    sim: SimulationCfg = SimulationCfg(
        dt=CONFIG['scene']['dt'],
        render_interval=CONFIG['scene']['render_interval']
    )
    
    # Scene config
    scene: SceneCfg = SceneCfg(
        num_envs=CONFIG['scene']['num_envs'],
        env_spacing=CONFIG['scene']['env_spacing'],
        replicate_physics=CONFIG['scene']['replicate_physics'],
        clone_in_fabric=CONFIG['scene']['clone_in_fabric'],
    )
    
    # Reward scale config
    rew_scale_grasp: float = CONFIG['reward']['scale_grasp']
    rew_scale_duration: float = CONFIG['reward']['scale_duration']
    rew_scale_distance: float = CONFIG['reward']['scale_distance']
    rew_scale_drop: float = CONFIG['reward']['scale_drop']
    rew_scale_contact: float = CONFIG['reward']['scale_contact']
    
    # Event config
    events: EventCfg = EventCfg()