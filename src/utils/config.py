from typing import Any
import yaml


def load_config(config: str) -> dict[str, Any]:
    '''
    Load a config from src/configs/yaml/
    
    Args:
        config (str): Config name; exclude file extension; assumed to be located in src/configs/yaml/
        
    Returns:
        dict[str, Any]: Loaded configuration
    '''
    # Read the config
    with open(f'src/configs/yaml/{config}.yaml', 'r') as f:
        return yaml.safe_load(f)