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
from src.utils.hyperparams import HPARAMS


thickness: float = HPARAMS['scene']['wall']['wall_thickness']


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
        spawn = sim_utils.DomeLightCfg(intensity=HPARAMS['scene']['light']['intensity'],
                                       color=HPARAMS['scene']['light']['color']),
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
        height=HPARAMS['scene']['sensor']['camera_height'],
        width=HPARAMS['scene']['sensor']['camera_width'],
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
        history_length=HPARAMS['scene']['sensor']['force_history_length'],
        debug_vis=True,
    )
    
    
    # Room borders
    dims = torch.rand((3,)) * (HPARAMS['scene']['wall']['wall_bounds'][1] - HPARAMS['scene']['wall']['wall_bounds'][0]) + HPARAMS['scene']['wall']['wall_bounds'][0]
    x_bound, y_bound, z_bound = get_rects(dims)
    
    # Generate and place prims
    wall_x1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallx1',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-float(dims[0].item()) / 2, 0.0, float(dims[2].item()) / 2),
        ),
    )
    wall_x2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallx2',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(dims[0].item()) / 2, 0.0, float(dims[2].item()) / 2)
        )
    )
    wall_y1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wally1',
        spawn=y_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -float(dims[1].item()) / 2, float(dims[2].item()) / 2)
        )
    )
    wall_y2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wally2',
        spawn=y_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, float(dims[1].item()) / 2, float(dims[2].item()) / 2)
        )
    )
    wall_z1 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallz1',
        spawn=z_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0)
        )
    )
    wall_z2 = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/wallz2',
        spawn=z_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(dims[2].item()))
        )
    )
    # Remove unnecessary objects prior to hashing
    del dims, x_bound, y_bound, z_bound