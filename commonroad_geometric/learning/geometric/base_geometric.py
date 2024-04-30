from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import dill
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import optimizer_to
from commonroad_geometric.common.utils.filesystem import load_dill
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.types import T_CommonRoadDataInput
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin

logger = logging.getLogger(__name__)


TypeVar_BaseGeometric = TypeVar('TypeVar_BaseGeometric', bound='BaseGeometric')

MODEL_FILENAME = 'model'
MODEL_FILETYPE = 'pt'
MODEL_FILE = Path(MODEL_FILENAME + '.' + MODEL_FILETYPE)
STATE_DICT_FILE = Path('state_dict.pt')
OPTIMIZER_FILE = Path('optimizer.pt')


class BaseModel(nn.Module, ABC, StringResolverMixin):

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).data.device

    def reset_parameters(self) -> None:
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class BaseGeometric(BaseModel):
    """Base class for facilitating training of GNNs, meant to
    handle all the overhead that goes into building and training.

    All subclasses must implement the following abstract methods:
        * compute_loss
        * _forward
        * configure_optimizer
        * _build

    Furthermore, the 'train_preprocess' method can be optionally implemented.

    See '/models/examples/' for a simple example of an implementation.
    """

    def __init__(self, cfg: Union[Config, dict]) -> None:
        if cfg is None:
            self.cfg = None
        else:
            self.cfg = cfg if isinstance(cfg, Config) else Config(cfg)
        self._latest_model_path: Optional[Path] = None
        self._latest_state_path: Optional[Path] = None
        self._optimizer: Optional[Optimizer] = None
        super(BaseGeometric, self).__init__()

    @abstractmethod
    def compute_loss(
        self,
        *outputs: Tensor,
        data: T_CommonRoadDataInput,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Computes and returns the loss for the current batch,
        given the forward output.

        Args:
            *outputs (Tensor): Outputs from model's 'forward' method.
            batch (Batch): Batch for which loss is computed
            epoch (int): Current training epoch, allowing loss configuration to change over time.
            **kwargs (Any): Loss configuration parameters

        Returns:
            Tuple[Tensor, Dict[str, Any]]:
            [1) Loss tensor, 2) info dictionary]

        Example:
        def compute_loss(
            self,
            z,
            data: CommonRoadData
            **kwargs
        ) -> Tuple[Tensor, Dict[str, Any]]:
            return z.sum(), {}
        """

    @abstractmethod
    def forward(
        self,
        data: T_CommonRoadDataInput,
        **kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Implementation of forward call.

        Args:
            data (T_CommonRoadDataInput): Current data

        Returns:
            Tensor or list of Tensors: Model output

        Example:

        def _forward(self, data: T_CommonRoadDataInput) ->  Union[Tensor, Tuple[Tensor, ...]]:
            return self.encoder(data)
        """

    def configure_optimizer(
        self,
        trial = None,
        optimizer_service = None,
    ) -> Optimizer:
        """Returns optimizer for the module.

        Args:
            trial: Optional[Trial]: Optuna Trial for the study being conducted if applicable

        Returns:
            Optimizer: PyTorch optimizer

        Example:
        def configure_optimizer(self) -> Optimizer:
            return torch.optim.RMSprop(
                self.parameters(),
                momentum=0.8
            )
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-3
        )

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRenderPlugin]]:
        """Returns list of renderer plugins for visualiasing the module's predictions.

        Returns:
            Optional[List[BaseRenderPlugin]]: List of renderer plugins, or None if visualization not implemented.
        """
        return None

    @abstractmethod
    def _build(
        self,
        batch: CommonRoadData,
        trial = None
    ) -> None:
        """Constructs all the submodules of the model. As opposed to the __init__ method,
        _build has access to data dimensions etc. via the CommonRoadData sample.

        Args:
            batch (CommonRoadData): Test batch for extracting data dimensions
            trial (Optional[Trial]): Optional parameter to include the trial from Optuna for adjusting model parameters

        def _build(self, batch: CommonRoadData, trial = None) -> None:
            self.linear = nn.Linear(
                batch.x.shape[1],
                1
            )
        """

    def train_preprocess(self, data: CommonRoadData) -> CommonRoadData:
        """Allows custom pre-processing of data instances during training.
        A possible application could be to apply Gaussian noise.

        Args:
            data (CommonRoadData): Incoming Data instance.

        Returns:
            CommonRoadData: Pre-processed Data instance.
        """
        return data

    def build(
        self,
        data: CommonRoadData,
        trial = None,
        optimizer_service = None,
        optimizer_state: Optional[Dict[str, Any]] = None
    ) -> None:
        self._trial = trial
        self._optimizer_service = optimizer_service
        if optimizer_state is None:
            self._build(data, trial)
        self._optimizer = self.configure_optimizer(trial=trial, optimizer_service=optimizer_service)
        if optimizer_state is not None:
            same_optimizer = True
            for i in range(len(self._optimizer.param_groups)):
                if set(self._optimizer.param_groups[i].keys()) != set(optimizer_state['param_groups'][i].keys()):
                    same_optimizer = False
                    break
            if same_optimizer:
                try:
                    self._optimizer.load_state_dict(optimizer_state)
                except ValueError as e:
                    logger.error(f'{type(self).__name__} failed to load optimizer state dict')
        optimizer_to(self._optimizer, data.device)

    @property
    def optimizer(self) -> Optimizer:
        if self._optimizer is None:
            raise AttributeError("self._optimizer is None")
        return self._optimizer  # type: ignore

    @property
    def latest_state_path(self) -> Optional[Path]:
        return self._latest_state_path

    @property
    def latest_model_path(self) -> Optional[Path]:
        return self._latest_model_path

    def save_model(
        self,
        output_path: Path,
        use_torch: bool = False
    ) -> None:
        if use_torch:
            torch.save(self, output_path, pickle_module=dill)
        else:
            with open(output_path, 'wb') as f:
                dill.dump(self, f)
        self._latest_model_path = Path(output_path)

    def save_state(
        self,
        output_path: Path
    ) -> None:
        torch.save(self.state_dict(), output_path)
        self._latest_state_path = output_path

    @classmethod
    def load(
        cls: Type[TypeVar_BaseGeometric],
        model_path: Path,
        device: Optional[str] = None,
        eval: bool = True,
        retries: int = 3,
        backoff_factor: float = 1.0,
        from_torch: bool = False,
        silent: bool = True
    ) -> TypeVar_BaseGeometric:

        exception = None
        for attempt in range(retries + 1):
            try:
                if from_torch:
                    model = torch.load(model_path, pickle_module=dill, map_location=device)
                else:
                    with open(model_path, 'rb') as f:
                        model = dill.load(f)
                model.to(device)
                if eval:
                    model.eval()
                break
            except Exception as e:
                # retrying in case of corrupt model file due to e.g. ongoing checkpoint saving
                if 'corrupt file' in str(e):
                    model = load_dill(model_path)
                    model.to(device)
                    if eval:
                        model.eval()
                    break
                if not silent:
                    logger.exception(
                        f"Failed to load BaseGeometric model from {model_path}. {retries - attempt - 1} attempts remaining.")
                wait_duration = backoff_factor * (2 ** (attempt - 1))
                exception = e
                if attempt < retries:
                    time.sleep(wait_duration)
        else:
            raise exception

        return model
