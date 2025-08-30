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
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path='/World/defaultGroundPlane',
        spawn=sim_utils.GroundPlaneCfg(),
    )
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
    