from typing import Any


def write_list(_list: list[Any], path: str) -> None:
    # Write to file
    with open(path, 'w') as f:
        f.write('\n'.join(_list))
        

def read_list(path: str) -> list[Any]:
    # Read file
    with open(path, 'r') as f:
        return f.read().strip().splitlines()