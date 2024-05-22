import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union


class LoggingFormat(Enum):
    VERBOSE = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    ONLY_FILENAMES = "%(asctime)s %(levelname)-7s %(filename)-30s %(message)s"
    NO_TIMESTAMP = "%(levelname)-7s %(filename)-30s %(message)s"
    NO_FILENAME = "%(asctime)-7s %(message)s"
    VERBOSE_MP = "pid=%(process)d, t=%(threadName)s - %(asctime)s %(name)s %(levelname)s: %(message)s"
    ONLY_FILENAMES_MP = "pid=%(process)-5d, t=%(threadName)-10s - %(asctime)s %(levelname)-7s %(filename)-30s %(message)s"
    NO_TIMESTAMP_MP = "pid=%(process)-5d, t=%(threadName)-10s - %(levelname)-7s %(filename)-30s %(message)s"
    NO_FILENAME_MP = "pid=%(process)-5d, t=%(threadName)-10s - %(levelname)-7s %(message)s"


class DeferredLogMessage(object):
    """
    Delayed __str__ conversion of expensive log messages
    """

    def __init__(self, lambda_msg: Callable[[Any], str]) -> None:
        self.lambda_msg = lambda_msg

    def __str__(self) -> str:
        return str(self.lambda_msg(None))


class BraceMessage:
    r"""
    Used to enable logging with f-string syntax. e.g. with {var}.

    References:
        https://docs.python.org/3/howto/logging-cookbook.html#formatting-styles
    """

    def __init__(self, fmt, /, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


def setup_logging(
    filename: Optional[Path] = None,
    level: Union[int, str] = logging.INFO,
    fmt: Union[LoggingFormat, str] = LoggingFormat.NO_FILENAME,
    mute_annoying_dependencies: bool = True
) -> None:
    root_logger = logging.getLogger()
    if isinstance(level, str):
        level = level.upper()
    root_logger.setLevel(level)
    formatter = logging.Formatter(
        fmt=fmt.value if isinstance(fmt, LoggingFormat) else fmt,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if mute_annoying_dependencies:
        logging.getLogger("urllib3").setLevel(logging.INFO)
        logging.getLogger("geopy").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        # TODO: CommonRoad?


def stdout_clear() -> None:
    sys.stdout.write('\033[2K\033[1G')


def stdout(s: Optional[Union[str, Sequence[str]]] = None) -> None:
    # TODO: Use args
    stdout_clear()
    sys.stdout.write(str(s) + "\r")
    sys.stdout.flush()


def set_terminal_title(title: Path, prefix_path: bool = True) -> None:
    if prefix_path:
        title = f"{sys.argv[0]} ({title})"
    title = f"{os.getlogin()}: {title}"
    print(f'\33]0;{title}\a', end='', flush=True)
