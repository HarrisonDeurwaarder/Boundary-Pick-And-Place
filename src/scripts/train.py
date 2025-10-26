import numpy as np
import torch
from torch.distributions import Uniform

from scripts.launch_app import launch_app

# Define and launch the app
sim_app, args_cli = launch_app()

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg

from sim.osc import update_states, get_osc, update_target, convert_to_task_frame
from utils.logger import logging
from configs.scene_cfg import SceneCfg


def run_sim(
    sim: sim_utils.SimulationContext, 
    scene: InteractiveScene,
    pose_dist: Uniform,
    wrench_dist: Uniform,
    kp_dist: Uniform,
) -> None:
    '''
    Runs the simulation
    
    Args:
        sim (SimulationContext): The simulation context
        scene (InteractiveScene): The interactive scene
        pose_dist (torch.distributions.Uniform): The uniform distribution to randomly sample target pose
        wrench_dist (torch.distributions.Uniform): The uniform distribution to randomly sample target wrench
        kp_dist (torch.distributions.Uniform): The uniform distribution to randomly sample target stiffness
    '''
    # Get indices for joints
    ee_frame_name = 'panda_leftfinger'
    arm_joint_names = ['panda_link.*']
    ee_frame_idx: int = scene['panda'].find_bodies(ee_frame_name)[0][0]
    arm_joint_ids: np.ndarray = scene['panda'].find_bodies(arm_joint_names)[0]
    
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
    count: int = 0
    while sim_app.is_running():
        # Episode reset procedure
        if count % 150 == 0:
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
            _, _, _, ee_pose_b, _, _, _, = update_states(
                panda,
                ee_frame_idx,
                arm_joint_ids,
            )
            
            # Sample the action
            ee_pose_task: torch.Tensor = pose_dist.sample((7,)).to(sim.device)
            ee_wrench_task: torch.Tensor = wrench_dist.sample((6,)).to(sim.device)
            kp_task: torch.Tensor = kp_dist.sample((6,)).to(sim.device)
            
            ee_targets: torch.Tensor = torch.cat([ee_pose_task, ee_wrench_task, kp_task])
            
            # Updates the command and specialized position/quaternion orientation target
            command, ee_target_pose_b = update_target(
                    sim,
                    scene,
                    osc,
                    ee_targets,
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
                arm_joint_ids=arm_joint_ids,
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
            panda.set_joint_effort_target(
                joint_efforts, joint_ids=arm_joint_ids,
            )
            panda.write_data_to_sim()

        # Perform step
        sim.step(render=True)
        # Update panda buffers
        panda.update(sim_dt)
        # Update scene buffers
        scene.update(sim_dt)
        # Update sim-time
        count += 1
        
        
def main() -> None:
    '''
    Main function to run the scene with OSC-calculated commands
    '''
    # Load the simulation
    sim_cfg: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1/120, render_interval=2, device=args_cli.device,)
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg,)
    # Design the scene and reset it
    scene_cfg: SceneCfg = SceneCfg(
        num_envs=9,
        env_spacing=20.0,
    )
    scene: InteractiveScene = InteractiveScene(scene_cfg,)
    sim.reset()
    # Log the completed setup
    logging.info('Setup complete.')
    
    run_sim(
        sim,
        scene,
        pose_dist=Uniform(-2.0, 2.0),
        wrench_dist=Uniform(0.0, 20.0),
        kp_dist=Uniform(0.0, 0.001)
    )
    

if __name__ == '__main__':
    main()
    sim_app.close()