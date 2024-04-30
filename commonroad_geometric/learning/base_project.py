from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union
import logging
from commonroad_geometric.debugging.profiling import profile


logger = logging.getLogger(__name__)


def register_run_command(func):
    def _decorator(self, *args, **kwargs):
        if self.cfg.profile:
            profile(partial(func, self=self), args=args, kwargs=kwargs)
        else:
            func(self, *args, **kwargs)
    _decorator.tagged = True
    return _decorator


class BaseProject(metaclass=ABCMeta):
    _commands: Dict[str, Callable[[], None]]

    def __init__(self) -> None:
        pass

    def __init_subclass__(cls) -> None:
        methods = dict((attr, getattr(cls, attr)) for attr in dir(cls) if not attr.startswith('__'))
        commands: Dict[str, Callable[[], None]] = {}
        for name, method in methods.items():
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, 'tagged'):
                commands[name] = method
        cls._commands = commands

    def run(self, cmd: str) -> None:
        subcommands = [s.strip() for s in cmd.split(' ')]
        subcommands = [s for s in subcommands if len(s) > 0]

        if not subcommands:
            exit_message = f'Please run the project using the command argument "run.py cmd=X". Available commands are:\n\n'
            for c in self._commands:
                exit_message += f" - {c}\n"
            logger.info(exit_message)
            return

        logger.info(f"Commands to be executed: {subcommands}")

        for subcmd in subcommands:
            if subcmd not in self._commands:
                raise ValueError(
                    f"Received undefined command '{subcmd}'. Available commands are: {list(self._commands.keys())}")

        for idx, subcmd in enumerate(subcommands):
            logger.info(f"Running command {idx+1}/{len(subcommands)}: '{subcmd}'")
            self._commands[subcmd](self)
