from typing import Any
import yaml


Params = dict[str, dict[str, float | int | bool | list[float]]]

# Read the hyperparameters
with open('configs/hyperparams.yaml', 'r') as f:
    HPARAMS: Params = yaml.safe_load(f)