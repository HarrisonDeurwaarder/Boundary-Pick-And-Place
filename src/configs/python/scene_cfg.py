import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

import torch
from src.utils.scene_setup import get_rects
from utils.config import load_config


CONFIG = load_config('panda_train')


thickness: float = CONFIG['scene']['room']['wall_thickness']
# Default wall spawn, prior to DR
default_spawn: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
    size=(1.0, 1.0, 1.0),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=True,
    ),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8, dynamic_friction=0.6, restitution=0.1,
    ),
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    '''
    Scene configuration
    '''
    # Ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path='/World/defaultGroundPlane',
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Lighting
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Light',
        spawn = sim_utils.DomeLightCfg(intensity=CONFIG['scene']['light']['intensity'],
                                       color=tuple(CONFIG['scene']['light']['color'])),
    )
    # Panda config
    panda: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path='{ENV_REGEX_NS}/Panda',
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, thickness),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path,
            activate_contact_sensors=True,
        )
    )
    
    # Sensors
    camera: CameraCfg = CameraCfg(
        prim_path='{ENV_REGEX_NS}/Panda/panda_hand/camera',
        update_period=0.1,
        height=CONFIG['scene']['sensor']['camera_height'],
        width=CONFIG['scene']['sensor']['camera_width'],
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            convention='ros'
        ),
    )
    
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Panda/panda_hand',
        update_period=0.0,
        history_length=CONFIG['scene']['sensor']['force_history_length'],
        debug_vis=True,
    )
    
    # Generate and place prims
    wall_x1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallx1',
        spawn=default_spawn,
    )
    wall_x2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallx2',
        spawn=default_spawn,
    )
    wall_y1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wally1',
        spawn=default_spawn,
    )
    wall_y2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wally2',
        spawn=default_spawn,
    )
    wall_z1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallz1',
        spawn=default_spawn,
    )
    wall_z2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallz2',
        spawn=default_spawn,
    )