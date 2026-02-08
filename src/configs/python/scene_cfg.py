import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_CFG

import torch
import src.utils.config as config


thickness: float = config.config['scene']['room']['wall_thickness']
# Default wall spawn, prior to DR
default_spawn: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
    size=(1.0, 1.0, 1.0),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=True,
    ),
    activate_contact_sensors=True,
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
        spawn = sim_utils.DomeLightCfg(intensity=config.config['scene']['light']['intensity'],
                                       color=tuple(config.config['scene']['light']['color'])),
    )
    # Robot config
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path='{ENV_REGEX_NS}/Robot',
        init_state=FRANKA_PANDA_CFG.init_state.replace(
            pos=(0.0, 0.0, thickness),
        ),
    )
    robot.spawn.activate_contact_sensors = True
    
    # Generate and place prims
    wall_x1 = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/wallx1',
        spawn=default_spawn,
    )
    wall_x2 = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/wallx2',
        spawn=default_spawn,
    )
    wall_y1 = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/wally1',
        spawn=default_spawn,
    )
    wall_y2 = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/wally2',
        spawn=default_spawn,
    )
    wall_z1 = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/wallz1',
        spawn=default_spawn,
    )
    wall_z2 = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/wallz2',
        spawn=default_spawn,
    )