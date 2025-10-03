import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

import torch
from utils.scene_setup import get_rects


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
    # Xform for all scene objects
    prim_utils.create_prim('/World/Objects', 'Xform')
    # Lighting
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path='/World/Light',
        spawn = sim_utils.DomeLightCfg(intensity=3000.0,
                                       color = (0.75, 0.75, 0.75)),
    )
    # Panda config
    panda: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path='{ENV_REGEX_NS}/Panda'
    )
    panda.actuators["panda_shoulder"].stiffness = 0.0
    panda.actuators["panda_shoulder"].damping = 0.0
    panda.actuators["panda_forearm"].stiffness = 0.0
    panda.actuators["panda_forearm"].damping = 0.0
    
    # Room borders
    dims = torch.rand((3,)) * 10 + 5
    x_bound, y_bound, z_bound = get_rects(dims)
    # Generate and place prims
    wall_x1 = RigidObjectCfg(
        prim_path='/World/Bounds/wallx1',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0)
        )
    )
    wall_x2 = RigidObjectCfg(
        prim_path='/World/Bounds/wallx2',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(dims[0].item()), 0.0, 0.0)
        )
    )
    wall_y1 = RigidObjectCfg(
        prim_path='/World/Bounds/wally1',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0)
        )
    )
    wall_y2 = RigidObjectCfg(
        prim_path='/World/Bounds/wally2',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, float(dims[1].item()), 0.0)
        )
    )
    wall_z1 = RigidObjectCfg(
        prim_path='/World/Bounds/wallz1',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0)
        )
    )
    wall_z2 = RigidObjectCfg(
        prim_path='/World/Bounds/wallz2',
        spawn=x_bound,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(dims[2].item()))
        )
    )
    # Remove unnecessary objects prior to hashing
    del dims, x_bound, y_bound, z_bound