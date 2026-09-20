import logging
from pathlib import Path
from datetime import datetime
import sys

from colorama import Fore, Style, init

init(autoreset=True)

LOGGER_NAME = "global"


class ColoredFormatter(logging.Formatter):
    TIME_COLOR = Fore.CYAN
    MESSAGE_COLOR = Fore.WHITE

    LEVEL_COLORS = {
        logging.DEBUG: Fore.MAGENTA,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)

        level_color = self.LEVEL_COLORS.get(record.levelno, "")

        if hasattr(record, "dict_log"):
            sep = record.dict_log.pop('__sep')
            message = record.dict_log.pop('__heading')
            parts = []

            max_length = max([len(key) for key in record.dict_log.keys()])

            for key, value in record.dict_log.items():
                _space = ''
                if sep == '\n':
                    _space = ' ' * (max_length - len(key))
                parts.append(
                    f"{Fore.YELLOW}{key + _space}{Style.RESET_ALL}: "
                    f"{Fore.BLUE}{value}{Style.RESET_ALL}"
                )

            message = message + "\n" + sep.join(parts)

        else:
            message = record.getMessage()

        return (
            f"{self.TIME_COLOR}{timestamp}{Style.RESET_ALL} "
            f"{level_color}[{record.levelname}]"
            f"{Style.RESET_ALL} "
            f"{message}"
        )


def configure_logging(log_level: str, log_to_terminal: bool) -> logging.Logger:

    logger = logging.getLogger(LOGGER_NAME)

    level = logging.getLevelNamesMapping()[log_level]

    msg_format = "%(asctime)s [%(levelname)s] %(message)s"

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"

    logger.setLevel(level)

    logger.handlers.clear()

    logger.propagate = False

    if log_to_terminal:
        stream_handler = logging.StreamHandler(sys.stdout)

        stream_handler.setFormatter(ColoredFormatter(msg_format))

        logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)

    file_handler.setFormatter(logging.Formatter(msg_format))

    logger.addHandler(file_handler)

    return logger


def dict_log(log_dict: dict, level: int, heading: str = "", sep = '\n'):
    logger = logging.getLogger(LOGGER_NAME)
    log_dict['__sep'] = sep
    log_dict['__heading'] = heading
    logger.log(
        level,
        heading + '\n' + sep.join([f'{key}: {value}' for key, value in log_dict.items()]),
        extra={"dict_log": log_dict},
    )
