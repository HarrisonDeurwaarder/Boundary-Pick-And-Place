import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    '''
    Scene configuration
    '''
    # Ground plane
    ground = AssetBaseCfg(
        prim_path='/World/defaultGroundPlane',
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Lighting
    light = AssetBaseCfg(
        prim_path='/World/Light',
        spawn = sim_utils.DomeLightCfg(intensity=3000.0,
                                       color = (0.75, 0.75, 0.75)),
    )
    # Panda config
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path='/World/envs/env_.*/Panda'
    )
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    