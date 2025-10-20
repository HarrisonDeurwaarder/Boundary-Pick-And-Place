from time import time

from utils.logger import logging
from typing import Callable, Any


def timer(func: Callable[..., Any],) -> Callable[..., Any]:
    '''
    Timer decorator, used to get run time of functions while debugging
    '''
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get the time prior to function call
        start = time()
        result: Any = func(*args, **kwargs)
        # Log the function run time
        logging.info(f'{func.__name__.title()} ran in {time() - start:.3f} seconds')
        return result
    return wrapper