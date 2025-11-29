from isaaclab.app import AppLauncher
from isaacsim import SimulationApp
import argparse
from typing import Any, Literal


Argument = dict[Literal['flag', 'type', 'default', 'help'], str | type | Any]

def launch_app(*args: Argument) -> tuple:
    '''
    Launch the app with the required flags
    
    Args:
        *args (dict[Literal['flag', 'type', 'default', 'help'], str | type | Any]): Dictionaries containing all required parameters of a argument to add to the parser
    
    Returns:
        tuple[SimulationApp, argpase.Namespace]: A tuple containing:
            - simulation_app: The launched app
            - args_cli: The arguments passed via CLI excecution
        
    '''
    # Define the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Pick-and-place Franka Panda using boundary points')
    for arg in args:
        parser.add_argument(
            arg['flag'],
            type=arg['type'],
            default=arg['default'],
            help=arg['help']
        )
    # Append and parse args
    AppLauncher.add_app_launcher_args(parser)
    args_cli: argparse.Namespace = parser.parse_args()
    
    # Launch IsaacSim with given args
    app_launcher: AppLauncher = AppLauncher(args_cli)
    simulation_app: SimulationApp = app_launcher.app
    
    return simulation_app, args_cli