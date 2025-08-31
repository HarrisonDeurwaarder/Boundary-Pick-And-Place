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
    sim['panda'].update(dt=sim_dt)
    
    # Center of panda's joint ranges
    joint_centers: torch.Tensor = torch.mean(scene['panda'].data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)
    
    # Get updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel,
    ) = update_states(
        panda=scene['panda'],
        ee_frame_idx=ee_frame_idx,
        arm_joint_ids=arm_joint_ids,
    )
    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)

    # Zero all joint efforts
    zero_joint_efforts = torch.zeros(scene.num_envs, scene['panda'].num_joints, device=sim.device)
    # The joint efforts touched by the OSC
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)
    
    
def update_states(panda: Articulation,
                  ee_frame_idx: int,
                  arm_joint_ids: list[int],) -> tuple[torch.Tensor]:
    '''
    Update the panda's states
    Contact forces are not explicitily handled for simplicity
    
    Args:
        panda (Articulation): The panda articulation
        ee_frame_idx (int): The end-effector frame index
        arm_joint_ids (list[int]): The arm joint indices
        
    Returns:
        jacobian_b (torch.Tensor): The jacobian in the body frame
        mass_mat (torch.Tensor): The mass matrix
        gravity (torch.Tensor): The gravity vector
        ee_pose_b (torch.tensor): The end-effector pose in the body frame
        ee_vel_b (torch.tensor): The end-effector velocity in the body frame
        root_pose_w (torch.tensor): The root pose in the world frame
        ee_pose_w (torch.tensor): The end-effector pose in the world frame
        joint_pos (torch.tensor): The joint positions
        joint_vel (torch.tensor): The joint velocities
    '''
    # Align frame indices with 0-start indexing
    ee_jacobi_idx: int = ee_frame_idx - 1
    # Obtain dynamics related quantities from the sim
    jacobian_w: torch.Tensor  = panda.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_mat: torch.Tensor = panda.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity: torch.Tensor = panda.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    
    # Convert the Jacobian from world to root frame
    jacobian_b: torch.Tensor = jacobian_w.clone()
    root_rot_matrix: torch.Tensor = matrix_from_quat(quat_inv(panda.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])
    
    # Compute current pose of the end_effector
    root_pos_w: torch.Tensor = panda.data.root_pos_w
    root_quat_w: torch.Tensor = panda.data.root_quat_w
    ee_pos_w: torch.Tensor = panda.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w: torch.Tensor = panda.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )
    root_pose_w: torch.Tensor = torch.cat(
        [root_pos_w, root_quat_w,],
        dim=-1,
    )
    ee_pose_w: torch.Tensor = torch.cat(
        [ee_pos_w, ee_quat_w,],
        dim=-1,
    )
    ee_pose_b: torch.Tensor = torch.cat(
        [ee_pos_b, ee_quat_b,],
        dim=-1,
    )
    
    # Extract EE/root vel in world frame
    ee_vel_w: torch.Tensor = panda.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w: torch.Tensor = panda.data.root_vel_w
    
    # Compute relative vel in world frame
    relative_vel_w: torch.Tensor = ee_vel_w - root_vel_w
    ee_lin_vel_b: torch.Tensor = quat_apply_inverse(panda.data.root_quat_w, relative_vel_w[:, :3])
    ee_ang_vel_b: torch.Tensor = quat_apply_inverse(panda.data.root_quat_w, relative_vel_w[:, 3:])
    ee_vel_b: torch.Tensor = torch.cat(
        [ee_lin_vel_b, ee_ang_vel_b],
        dim=-1,
    )
    
    # Get joint positions and velocities
    joint_pos: torch.Tensor = panda.data.joint_pos[:, arm_joint_ids]
    joint_vel: torch.Tensor = panda.data.joint_vel[:, arm_joint_ids]
    
    return (
        jacobian_b,
        mass_mat,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel
    )