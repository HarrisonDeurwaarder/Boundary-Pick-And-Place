import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene


def get_osc(sim: sim_utils.SimulationContext,
            scene: InteractiveScene,) -> OperationalSpaceController:
    '''
    Instantiate the OSC and return it
    
    Args:
        sim (SimulationContext): The simulation context
        scene (InteractiveScene): The interactive scene
    '''
    # Construct the OSC
    cfg: OperationalSpaceControllerCfg = OperationalSpaceControllerCfg(
        target_types=['pose_abs', 'wrench_abs'],
        impedance_mode='variable_kp',
        inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
        nullspace_control='position',
    )
    osc: OperationalSpaceController = OperationalSpaceController(
        cfg, 
        num_envs=scene.num_envs,
        device=sim.device
    )
    return osc


def run_sim(sim: sim_utils.SimulationContext, 
            scene: InteractiveScene,) -> None:
    '''
    Runs the simulation
    
    Args:
        sim (SimulationContext): The simulation context
        scene (InteractiveScene): The interactive scene
    '''
    # Get indices for joints
    ee_frame_idx: int = scene['robot'].find_bodies('panda_leftfinger')[0][0]
    arm_joint_ids: np.ndarray = scene['robot'].find_bodies(['panda_joint.*'])[0]
    
    # Define the OSC
    osc: OperationalSpaceController = get_osc(sim, scene,)
    
    sim_dt: float = sim.get_physics_dt()
    sim['panda']