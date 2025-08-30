import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)


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
    
    
def update_states(sim: sim_utils.SimulationContext,
                  scene: InteractiveScene,
                  panda: Articulation,
                  ee_frame_idx: int,
                  arm_joint_ids: list[int],
                  contact_forces: torch.Tensor,) -> tuple:
    '''
    Update the panda's states
    
    Args:
        sim (SimulationContext): The simulation context
        scene (InteractiveScene): The interactive scene
        panda (Articulation): The panda articulation
        ee_frame_idx (int): The end-effector frame index
        arm_joint_ids: (list[int]) The arm joint indices
        
    Returns:

    '''
    # Align frame indices with 0-start indexing
    ee_jacobi_idx = ee_frame_idx - 1
    # Obtain dynamics related quantities from the sim
    jacobian_w  = panda.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_mat = panda.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = panda.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(panda.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])
    
    # Compute current pose of the end_effector
    root_pos_w = panda.data.root_pos_w
    root_quat_w = panda.data.root_quat_w
    ee_pos_w = panda.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = panda.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )
    root_pose_w = torch.cat(
        [root_pos_w, root_quat_w,],
        dim=-1,
    )
    ee_pose_w = torch.cat(
        [ee_pos_w, ee_quat_w,],
        dim=-1,
    )
    ee_pose_b = torch.cat(
        [ee_pos_b, ee_quat_b,],
        dim=-1,
    )
    
    # Extract EE/root vel in world frame
    ee_vel_w = panda.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w = panda.data.root_vel_w
    # Compute relative vel in world frame
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(panda.data)