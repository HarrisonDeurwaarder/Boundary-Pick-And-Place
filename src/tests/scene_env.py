import numpy as np
import torch
from torch.distributions import Uniform

from src.sim.launch_app import launch_app

# Define and launch the app
sim_app, args_cli = launch_app(
    enable_cameras=True,
)

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from src.sim.osc import update_states, get_osc, update_target, convert_to_task_frame
from src.utils.logger import logging
from src.utils.hyperparams import HPARAMS
from src.configs.scene_cfg import SceneCfg
from src.configs.environment_cfg import EnvCfg
from src.rl.environment import Env


def run_sim(
    sim: sim_utils.SimulationContext,
    env: Env,
    pose_dist: Uniform,
    wrench_dist: Uniform,
    kp_dist: Uniform,
) -> None:
    '''
    Runs the simulation
    
    Args:
        sim (SimulationContext): The simulation context
        env (Env): The RL environment
        pose_dist (torch.distributions.Uniform): The uniform distribution to randomly sample target pose
        wrench_dist (torch.distributions.Uniform): The uniform distribution to randomly sample target wrench
        kp_dist (torch.distributions.Uniform): The uniform distribution to randomly sample target stiffness
    '''
    scene: InteractiveScene = env.scene
    # Get indices for joints
    ee_frame_name = 'panda_leftfinger'
    arm_joint_names = ['panda_link.*']
    ee_frame_idx: int = scene['panda'].find_bodies(ee_frame_name)[0][0]
    arm_joint_ids: np.ndarray = scene['panda'].find_bodies(arm_joint_names)[0]
    
    # Define the OSC
    osc: OperationalSpaceController = get_osc(sim, scene,)
    
    sim_dt: float = sim.get_physics_dt()
    contact_forces: ContactSensorCfg = scene['contact_forces']
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
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = update_states(
        sim=sim,
        scene=scene,
        panda=panda,
        ee_frame_idx=ee_frame_idx,
        arm_joint_ids=arm_joint_ids,
        contact_forces=contact_forces,
    )
    
    command: torch.Tensor = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b: torch.Tensor = torch.zeros(scene.num_envs, 7, device=sim.device)
    
    osc.set_command(
        command=command,
        current_ee_pose_b=ee_pose_b,
        current_task_frame_pose_b=ee_pose_b,
    )
    
    # Initial environment reset
    env.reset()
    
    # The joint efforts touched by the OSC
    joint_efforts: torch.Tensor = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    ''' Simulation Loop '''
    count: int = 0
    while sim_app.is_running():
        (
            jacobian_b,
            mass_mat,
            gravity,
            ee_pose_b, 
            ee_vel_b,
            ee_force_b,
            joint_pos,
            joint_vel
        ) = update_states(
            sim=sim,
            scene=scene,
            panda=panda,
            ee_frame_idx=ee_frame_idx,
            arm_joint_ids=arm_joint_ids,
            contact_forces=contact_forces,
        )
        # Get joint commands
        joint_efforts: torch.Tensor = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_mat,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )
        # Apply environment step
        _, _, term, trunc, _ = env.step(joint_efforts)
        
        # Perform step
        sim.step(render=True)
        # Update scene buffers
        scene.update(sim_dt)
        # Update sim-time
        count += 1
        
        # If episode has ended
        if torch.any(term) or torch.any(trunc):
            # Call reset
            env.reset()
            # Update initial states
            _, _, _, ee_pose_b, _, _, _, _ = update_states(
                sim=sim,
                scene=scene,
                panda=panda,
                ee_frame_idx=ee_frame_idx,
                arm_joint_ids=arm_joint_ids,
                contact_forces=contact_forces,
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
            logging.info('Environment reset.')

        
        
def main() -> None:
    '''
    Main function to run the scene with OSC-calculated commands
    '''
    # Create the environment
    env_cfg: EnvCfg = EnvCfg()
    env: Env = Env(env_cfg)
    # Load the simulation
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(env_cfg.sim,)
    sim.reset()
    # Log the completed setup
    logging.info('Setup complete.')
    
    run_sim(
        sim,
        env,
        pose_dist=Uniform(-2.0, 2.0),
        wrench_dist=Uniform(0.0, 20.0),
        kp_dist=Uniform(0.0, 0.001)
    )
    

if __name__ == '__main__':
    main()
    sim_app.close()