import torch

from source.sim.launch_app import launch_app
from source.utils.config import load_config

sim_app, args_cli = launch_app(
    enable_cameras=True,
)
load_config('train')
    
from source.configs.python.scene_cfg import SceneCfg
from source.utils.logger import logging
    
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg


def main() -> None:
    '''
    Main function for scene rendering tests
    '''
    # Load the simulation
    sim_cfg: SimulationCfg = SimulationCfg(dt=0.01, device=args_cli.device,)
    sim: SimulationContext = SimulationContext(sim_cfg,)
    # Design the scene
    scene_cfg: SceneCfg = SceneCfg(
        num_envs=9,
        env_spacing=1.0,
    )
    scene: InteractiveScene = InteractiveScene(scene_cfg,)
    sim.reset()
    logging.info('Setup complete.')
    
    sim_dt: float = sim.get_physics_dt()
    robot = scene['robot']
    # Update robot buffers prior to first step
    robot.update(dt=sim_dt,)
    # Reset the robot to default states
    default_joint_pos, default_joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel,)
    robot.set_joint_effort_target(torch.zeros(
        scene.num_envs, 
        robot.num_joints,
        device=sim.device,
    ),)
    robot.write_data_to_sim()
    robot.reset()
    
    # Play the sim
    while sim_app.is_running():
        # Write scene and robot data
        robot.write_data_to_sim()
        scene.write_data_to_sim()
        
        # Update the scene
        sim.step(render=True)
        robot.update(sim_dt,)
        scene.update(sim_dt,)
        
        
if __name__ == '__main__':
    main()
    sim_app.close()