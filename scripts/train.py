import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation

from sim.osc import update_states, get_osc, update_target, convert_to_task_frame
from sim.loop import terminate, base
from scripts.launch_app import launch_app


def run_sim(sim: sim_utils.SimulationContext, 
            scene: InteractiveScene,) -> None:
    '''
    Runs the simulation
    
    Args:
        sim (SimulationContext): The simulation context
        scene (InteractiveScene): The interactive scene
    '''
    # Define and launch the app
    sim_app, args_cli = launch_app()
    
    # Get indices for joints
    ee_frame_idx: int = scene['robot'].find_bodies('panda_leftfinger')[0][0]
    arm_joint_ids: np.ndarray = scene['robot'].find_bodies(['panda_joint.*'])[0]
    
    # Define the OSC
    osc: OperationalSpaceController = get_osc(sim, scene,)
    
    sim_dt: float = sim.get_physics_dt()
    panda: Articulation = scene['panda']
    panda.update(dt=sim_dt)
    
    # Center of panda's joint ranges
    joint_centers: torch.Tensor = torch.mean(panda.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)
    
    # Get updated states
    (
        jacobian_b,
        mass_mat,
        gravity,
        ee_pose_b,
        ee_vel_b,
        joint_pos,
        joint_vel,
    ) = update_states(
        panda=panda,
        ee_frame_idx=ee_frame_idx,
        arm_joint_ids=arm_joint_ids,
    )
    
    command: torch.Tensor = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b: torch.Tensor = torch.zeros(scene.num_envs, 7, device=sim.device)

    # Zero all joint efforts
    zero_joint_efforts: torch.Tensor = torch.zeros(scene.num_envs, panda.num_joints, device=sim.device)
    # The joint efforts touched by the OSC
    joint_efforts: torch.Tensor = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    '''Simulation Loop'''
    terminated: bool = True
    while sim_app.is_running():
        # Episode reset procedure
        if terminated:
            # Reset joint pos to default
            # Temporary for simplicity (will be randomized)
            default_joint_pos: torch.Tensor = panda.data.default_joint_pos.clone()
            default_joint_vel: torch.Tensor = panda.data.default_joint_vel.clone()
            panda.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            panda.set_joint_effort_target(zero_joint_efforts)
            panda.write_data_to_sim()
            panda.reset()
            # Reset target pose
            panda.update(sim_dt)
            _, _, _, ee_pose_b, _, _, _, _, _ = update_states(
                panda,
                ee_frame_idx,
                arm_joint_ids
            )
            # Updates the command and specialized position/quaternion orientation target
            command, ee_target_pose_b = update_target(
                    osc, _
                )
            # Set the OSC command
            osc.reset()
            command, task_frame_pose_b = convert_to_task_frame(
                osc,
                command,
                ee_target_pose_b,
            )
            osc.set_command(
                command=command,
                current_ee_pose_b=ee_pose_b,
                current_task_frame_pose_b=task_frame_pose_b,
            )
        # Base execution; steps sim
        else:
            (
                jacobian_b,
                mass_mat,
                gravity,
                ee_pose_b,
                ee_vel_b,
                joint_pos,
                joint_vel
            ) = update_states(
                panda=panda,
                ee_frame_idx=ee_frame_idx,
                ee_target_pose_b=ee_target_pose_b,
            )
            # Get joint commands
            joint_efforts: torch.Tensor = osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                mass_matrix=mass_mat,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            # Apply efforts
            scene['panda'].set_joint_effort_target(
                joint_efforts, joint_ids=arm_joint_ids,
            )
            scene['panda'].write_data_to_sim()

        # Perform step
        sim.step(render=True)
        # Update robot buffers
        scene['panda'].update(sim_dt)
        # Update scene buffers
        scene.update(sim_dt)
        # Update sim-time
        count += 1