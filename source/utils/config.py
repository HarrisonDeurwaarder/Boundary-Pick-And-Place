from typing import Any
import yaml


config: dict[str, Any] | None = None


def load_config(filename: str) -> None:
    '''
    Sets the current configuration
    
    Args:
        filename (str): Config file name; exclude file extension; assumed to be located in src/configs/yaml/
    '''
    global config
    # Read the config
    with open(f'src/configs/yaml/{filename}.yaml', 'r') as f:
        config = yaml.safe_load(f)