from __future__ import annotations
from commonroad_geometric.common.utils.datetime import get_timestamp
from commonroad_geometric.learning.training.wandb_service.constants import (
    PROJECT_NAME,
    ENTITY_NAME
)
from commonroad_geometric.common.logging import set_terminal_title
from typing import Dict, Any, Optional, TYPE_CHECKING
import socket
import os
import logging
from wandb import sdk as wandb_sdk
import wandb

from dotenv import load_dotenv
load_dotenv()
if TYPE_CHECKING:
    from commonroad_geometric.learning.geometric.training.types import GeometricTrainingContext


logger = logging.getLogger(__name__)


os.environ["WANDB_START_METHOD"] = "thread"


class WandbService:
    def __init__(
        self,
        entity_name: Optional[str] = None,
        project_name: Optional[str] = None,
        *,
        disable: bool = False,
    ) -> None:
        """ Initializes wandb using the api key present in the environment
        """
        self._watch = False
        self._entity_name = entity_name if entity_name is not None else os.environ.get(ENTITY_NAME)
        self._project_name = project_name if project_name is not None else os.environ.get(PROJECT_NAME)
        self._disable = disable
        self._experiment_name: Optional[str] = None
        if self._disable:
            self._success = False
        else:
            if self._project_name is None:
                self._success = False
                logger.warn(
                    f"wandb integration not configured yet (PROJECT_NAME={self._project_name}, ENTITY_NAME={self._entity_name}).")
            else:
                self._success = True
                logger.info(
                    f"Successfully retrieved wandb environment configuration (PROJECT_NAME={self._project_name}, ENTITY_NAME={self._entity_name}).")
        self._current_run = None

    def start_experiment(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        include_timestamp: bool = True,
        include_random_name: bool = True,
        include_hostname: bool = True,
        update_terminal_title: bool = True,
        **init_kwargs: Any
    ) -> str:
        if not self._success:
            return name
        metadata = metadata if metadata is not None else {}
        wandb.config = metadata
        name = f"{name}-{get_timestamp()}" if include_timestamp else name
        name = f"{socket.gethostname()}-{name}" if include_hostname else name

        try:
            self._current_run = wandb.init(
                project=self._project_name,
                entity=self._entity_name,
                config=metadata,
                **init_kwargs
            )
            name = name + '-' + wandb.run.name if include_random_name else name
            logger.info(f"Starting wandb experiment '{name}'.")
            wandb.run.name = name
            # wandb.run.save()
            self._success = True
            if update_terminal_title:
                set_terminal_title(name)
        except Exception:
            self._success = False

        self._experiment_name = name
        return name

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def success(self) -> bool:
        return self._success

    @property
    def disabled(self) -> bool:
        return self._disable

    @property
    def config(self) -> wandb_sdk.wandb_config.Config:
        return wandb.config

    def update_config(self, attributes: Dict[str, list]) -> bool:
        return wandb.config.update(attributes)

    def define_metric(self, metric_name: str, step_metric: str = None):
        if self._success:
            if step_metric is None:
                wandb.define_metric(metric_name)
            else:
                # Allows defining a custom x-axis against which metrics can be logged
                wandb.define_metric(metric_name, step_metric=step_metric)

    def watch_model(self, ctx: GeometricTrainingContext, log_freq: int, **kwargs) -> None:
        if not self._watch:
            wandb.watch(ctx.model, log_freq=log_freq, **kwargs)
            self._watch = True

    def finish_experiment(self) -> None:
        if self._current_run is not None:
            self._current_run.finish()  # TODO: context manager

    def log(self, msg_dict: Dict[str, Any], step=None) -> None:
        if self._success:
            if step is not None:
                wandb.log(msg_dict, step=step)
            else:
                wandb.log(msg_dict)
