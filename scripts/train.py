import numpy as np
import torch

from utils.osc import *

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene


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
