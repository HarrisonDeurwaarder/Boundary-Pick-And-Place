import logging
import time


class Formatter(logging.Formatter):
    '''
    Formats logging using time elapsed instead of timestamp
    '''
    def __init__(self, 
                 fmt: str | None = None,) -> None:
        super().__init__(fmt,)
        self.start_time = time.time()
        
    def format(self, 
               record: logging.LogRecord,) -> str:
        # Calculate and format time elapsed
        elapsed = time.time() - self.start_time
        record.elapsed = f'{elapsed:.2f}'
        
        return super().format(record)
    
    
# Setup logging
handler: logging.StreamHandler = logging.StreamHandler()
formatter: Formatter = Formatter('[%(elapsed)s][%(levelname)s]: %(message)s',)
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])