
import logging
from pathlib import Path
from datetime import datetime

LOGGER_NAME = "global"

logger = logging.Logger(LOGGER_NAME)
level = logging.DEBUG
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)

logger.setLevel(level)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)