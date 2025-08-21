from isaaclab.app import AppLauncher
import argparse
from utils.logger import logging


def launch() -> tuple:
    '''
    Launch the app with the required flags
    '''
    # Define the argument parser
    parser = argparse.ArgumentParser(description='[ADD LATER]')
    # Append and parse args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    # Launch IsaacSim with given args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    from isaaclab.sim import SimulationCfg, SimulationContext

    # Configure sim context
    sim_cfg = SimulationCfg(dt=0.01)
    sim_context = SimulationContext(sim_cfg)
    logging.info('App successfully launched')
    
    return simulation_app, sim_context, args_cli


if __name__ == '__main__':
    launch()
    logging.info('success')