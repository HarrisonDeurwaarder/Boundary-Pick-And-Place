import torch

from scripts.launch_app import launch_app

sim_app, args_cli = launch_app()
    
from configs.scene_cfg import SceneCfg
from utils.logger import logging
    
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
    # Update panda buffers prior to first step
    scene['panda'].update(dt=sim_dt,)
    # Reset the panda to default states
    default_joint_pos, default_joint_vel = scene['panda'].data.default_joint_pos.clone(), scene['panda'].data.default_joint_vel.clone()
    scene['panda'].write_joint_state_to_sim(default_joint_pos, default_joint_vel,)
    scene['panda'].set_joint_effort_target(torch.zeros(
        scene.num_envs, 
        scene['panda'].num_joints,
        device=sim.device,
    ),)
    scene['panda'].write_data_to_sim()
    scene['panda'].reset()
    
    # Play the sim
    while sim_app.is_running():
        # Write scene and panda data
        scene['panda'].write_data_to_sim()
        scene.write_data_to_sim()
        
        # Update the scene
        sim.step(render=True)
        scene['panda'].update(sim_dt,)
        scene.update(sim_dt,)
        
        
if __name__ == '__main__':
    main()
    sim_app.close()