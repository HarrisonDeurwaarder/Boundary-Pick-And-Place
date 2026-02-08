import isaaclab.sim as sim_utils

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg

from src.configs.python.scene_cfg import SceneCfg
from src.configs.python.event_cfg import EventCfg
import src.utils.config as config


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
    
    # Reward scale config
    rew_scale_grasp: float = config.config['reward']['scale_grasp']
    rew_scale_duration: float = config.config['reward']['scale_duration']
    rew_scale_distance: float = config.config['reward']['scale_distance']
    rew_scale_drop: float = config.config['reward']['scale_drop']
    rew_scale_contact: float = config.config['reward']['scale_contact']
    
    # Sensors to be injected into scene
    camera: CameraCfg = CameraCfg(
        prim_path='/World/envs/env_.*/Robot/panda_hand/front_cam',
        update_period=0.1,
        height=config.config['scene']['sensor']['camera']['camera_height'],
        width=config.config['scene']['sensor']['camera']['camera_width'],
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=config.config['scene']['sensor']['camera']['focal_length'],
            focus_distance=config.config['scene']['sensor']['camera']['focus_distance'],
            horizontal_aperture=config.config['scene']['sensor']['camera']['horizontal_aperture'],
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            convention='ros'
        ),
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path='/World/envs/env_.*/Robot/panda_hand',
        update_period=0.0,
        history_length=config.config['scene']['sensor']['force_history_length'],
        debug_vis=True,
    )
    
    # Event config
    events: EventCfg = EventCfg()