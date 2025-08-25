from scripts.launch_app import launch_app

sim_app, args_cli = launch_app()
    
from configs.scene_cfg import SceneCfg
from utils.logger import logging
    
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg


def main():
    '''
    Main function for scene rendering tests
    '''
    # Load the simulation
    sim_cfg: SimulationCfg = SimulationCfg(device=args_cli.device,)
    sim: SimulationContext = SimulationContext(sim_cfg,)
    # Design the scene
    scene_cfg: SceneCfg = SceneCfg(
        num_envs=1,
        env_spacing=1.0,
    )
    scene = InteractiveScene(scene_cfg,)
    logging.info('Setup complete.')
    
    # Play the sim
    sim_dt = sim.get_physics_dt()
    # Update panda buffers prior to first step
    scene['robot'].update(sim_dt)
    while sim_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        
        scene['robot'].update(sim_dt)
        scene.update(sim_dt,)
        
        
if __name__ == '__main__':
    main()
    sim_app.close()