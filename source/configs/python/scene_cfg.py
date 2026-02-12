import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, MassPropertiesCfg

from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import torch
import os
from pathlib import Path

from random import shuffle
import source.utils.config as config
from source.utils.file_io import read_list


rb_usd_files: list[str] = read_list('source/data/rb_usd_paths.txt')
usd_assets: list[UsdFileCfg] = [sim_utils.UsdFileCfg(usd_path=path,) for path in rb_usd_files[:50]]


thickness: float = config.config['scene']['room']['wall_thickness']
# Default wall spawn, prior to DR
'''default_spawn: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
    size=(1.0, 1.0, 1.0),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=True,
    ),
    activate_contact_sensors=True,
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0),),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8, dynamic_friction=0.6, restitution=0.1,
    ),
)'''

default_object_spawn: sim_utils.MultiAssetSpawnerCfg = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=usd_assets,
    random_choice=True,
)
# Dimension range
dim_range: list = torch.arange(0.04, 0.1, 0.02).tolist()


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
    
    # Sensors to be injected into scene
    camera: TiledCameraCfg = TiledCameraCfg(
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
        filter_prim_paths_expr=['{ENV_REGEX_NS}/object_' + str(i) for i in range(config.config['scene']['object']['n_assets'])]
    )
    '''
    # Generate and place prims
    wall_x1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallx1',
        spawn=default_spawn.replace(),
    )
    wall_x2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallx2',
        spawn=default_spawn.replace(),
    )
    wall_y1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wally1',
        spawn=default_spawn.replace(),
    )
    wall_y2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wally2',
        spawn=default_spawn.replace(),
    )
    wall_z1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallz1',
        spawn=default_spawn.replace(),
    )
    wall_z2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallz2',
        spawn=default_spawn.replace(),
    )'''
    
# Objects to pick
'''for i in range(config.config['scene']['object']['n_assets']):
    setattr(SceneCfg, f'object_{i}', RigidObjectCfg(
        prim_path=f'/World/envs/env_.*/object_{i}',
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=cone_radius,
                    height=cone_height,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ) for cone_radius in dim_range for cone_height in dim_range for _ in dim_range
            ] + [
                sim_utils.CuboidCfg(
                    size=(x, y, z),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ) for x in dim_range for y in dim_range for z in dim_range
            ] + [
                sim_utils.SphereCfg(
                    radius=sphere_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ) for sphere_radius in dim_range  for _ in dim_range for _ in dim_range
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=1.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    ))
    '''
for i in range(config.config['scene']['object']['n_assets']):
    setattr(SceneCfg, f'object_{i}', RigidObjectCfg(
        prim_path=f'/World/envs/env_.*/object_{i}',
        spawn=default_object_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    ))