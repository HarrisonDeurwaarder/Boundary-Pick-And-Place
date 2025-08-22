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
    
    return simulation_app, args_cli