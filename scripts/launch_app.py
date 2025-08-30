from isaaclab.app import AppLauncher
from isaacsim import SimulationApp
import argparse
from utils.logger import logging


def launch_app() -> tuple:
    '''
    Launch the app with the required flags
    
    Returns:
        simulation_app (SimulationApp): The launched app
        args_cli (argparse.Namespace): The arguments passed via CLI excecution
        
    '''
    # Define the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='[ADD LATER]')
    # Append and parse args
    AppLauncher.add_app_launcher_args(parser)
    args_cli: argparse.Namespace = parser.parse_args()
    
    # Launch IsaacSim with given args
    app_launcher: AppLauncher = AppLauncher(args_cli)
    simulation_app: SimulationApp = app_launcher.app
    
    return simulation_app, args_cli