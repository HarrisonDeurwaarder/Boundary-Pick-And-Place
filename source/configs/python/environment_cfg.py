import isaaclab.sim as sim_utils

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg

from source.configs.python.scene_cfg import SceneCfg
from source.configs.python.event_cfg import EventCfg
import source.utils.config as config


@configclass
class EnvCfg(DirectRLEnvCfg):
    '''
    Environment configuration
    '''
    # Env config
    decimation: int = config.config['env']['decimation']
    episode_length_s: float = config.config['env']['episode_length'] # [sec]
    action_scale: float = config.config['env']['action_scale']
    action_space: float = config.config['env']['action_space']
    observation_space: int = config.config['env']['angle_obs_space'] + config.config['env']['velocity_obs_space']
    state_space: int = config.config['env']['state_space']
    
    # Sim config
    sim: SimulationCfg = SimulationCfg(
        dt=config.config['scene']['dt'],
        render_interval=config.config['scene']['render_interval'],
    )
    
    # Scene config
    scene: SceneCfg = SceneCfg(
        num_envs=config.config['scene']['num_envs'],
        env_spacing=config.config['scene']['env_spacing'],
        replicate_physics=config.config['scene']['replicate_physics'],
        clone_in_fabric=config.config['scene']['clone_in_fabric'],
    )
    
    # Event config
    events: EventCfg = EventCfg()