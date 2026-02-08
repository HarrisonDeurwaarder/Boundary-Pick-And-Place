import torch
from torch.distributions import Uniform

from src.sim.launch_app import launch_app
from src.utils.config import load_config

# Define and launch the app
sim_app, args_cli = launch_app(
    enable_cameras=True,
)

load_config('train')

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
import isaacsim.core.utils.stage as stage_utils

from src.sim.osc import update_states, get_osc, update_target, convert_to_task_frame
from src.utils.logger import logging
from src.configs.python.scene_cfg import SceneCfg
from src.configs.python.environment_cfg import EnvCfg
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
    
    sim_dt: float = sim.get_physics_dt()
    robot: Articulation = scene['robot']
    robot.update(dt=sim_dt)
    
    # Initial environment reset
    env.reset()

    ''' Simulation Loop '''
    count: int = 0
    while sim_app.is_running():
        
        # Sample the action
        ee_pose_task: torch.Tensor = pose_dist.sample((env.num_envs, 7,)).to(sim.device)
        ee_wrench_task: torch.Tensor = wrench_dist.sample((env.num_envs, 6,)).to(sim.device)
        kp_task: torch.Tensor = kp_dist.sample((env.num_envs, 6,)).to(sim.device)
        
        ee_targets: torch.Tensor = torch.cat([ee_pose_task, ee_wrench_task, kp_task], dim=1)
        # Environment takes an ee target, each decimation a substep is then taken
        _, _, term, trunc, _ = env.step(ee_targets)
        
        # Perform step
        sim.step(render=True)
        # Update scene buffers
        scene.update(sim_dt)
        # Update sim-time
        count += 1

        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["contact_forces"])
        print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())
        
        # If episode has ended
        if torch.any(term) or torch.any(trunc):
            # Call reset
            env.reset()
            
            logging.info('Environment reset.')

        
        
def main() -> None:
    '''
    Main function to run the scene with OSC-calculated commands
    '''
    
    # Create the environment
    env_cfg: EnvCfg = EnvCfg()
    env: Env = Env(env_cfg)
    # Load the simulation
    '''sim: sim_utils.SimulationContext = sim_utils.SimulationContext(env_cfg.sim,)
    sim.reset()'''
    env.reset()
    # Log the completed setup
    logging.info('Setup complete.')
    
    run_sim(
        env.sim,
        env,
        pose_dist=Uniform(-2.0, 2.0),
        wrench_dist=Uniform(0.0, 20.0),
        kp_dist=Uniform(0.0, 0.001)
    )
    

if __name__ == '__main__':
    main()
    sim_app.close()