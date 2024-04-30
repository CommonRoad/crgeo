import logging
from typing import Iterable, Optional, Set
from pathlib import Path
from typing import List, Dict, Tuple, TypeVar
from itertools import chain

import torch
from commonroad_geometric.common.progress_reporter import ProgressReporter
from commonroad_geometric.dataset.transformation.dataset_transformation import dataset_transformation
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from sklearn.preprocessing import StandardScaler
from torch import Tensor

logger = logging.getLogger(__name__)

NormalizationParams = Tuple[Tensor, Tensor]
T_CommonRoadData = TypeVar("T_CommonRoadData", CommonRoadData, CommonRoadDataTemporal)


def get_normalization_params_file_path(processed_dir: Path) -> Path:
    return processed_dir / "../normalization_params.pt"


class FeatureNormalizer:

    def __init__(self, ignore_keys: Optional[Set[Tuple[str, str]]] = None):
        self._ignore_keys = ignore_keys if ignore_keys is not None else set()
        self._scalers: Dict[str, StandardScaler] = {}
        self._params: Dict[str, NormalizationParams] = {}
        self._initialized: bool = False

    def init_scalers(self, data: T_CommonRoadData):
        self._initialized = True
        for store in chain(data.node_stores, data.edge_stores):
            for key in store.keys(virtual_attributes=False):
                attr_tuple = (store.key, key)
                if attr_tuple in self._ignore_keys:
                    continue
                if self.is_feature(data, store=store.key, key=key):
                    self._scalers[attr_tuple] = StandardScaler()
                    logger.info(f"Created StandardScaler for feature {attr_tuple}")

    @staticmethod
    def is_feature(data: T_CommonRoadData, store: str, key: str) -> bool:
        tensor = data[store][key]
        is_feature = isinstance(tensor, Tensor) and torch.is_floating_point(
            tensor) and tensor.ndim <= 2 and tensor.size(-1) > 0
        return is_feature

    def partial_fit(self, data: T_CommonRoadData) -> None:
        # [k for k in data.v.keys(virtual_attributes=False)]
        if not self._initialized:
            self.init_scalers(data)

        for (store, key), scaler in self._scalers.items():
            if data[store][key].size(0) == 0:
                continue
            value = data[store][key].numpy()
            scaler.partial_fit(value)

    def prepare_parameters(self) -> None:
        for attr_key, scaler in self._scalers.items():
            assert hasattr(scaler, "n_samples_seen_") and scaler.n_samples_seen_ > 0, f"{attr_key} has not been fitted"
            self._params[attr_key] = (
                torch.from_numpy(scaler.mean_).type(torch.float32),  # mean
                1.0 / torch.from_numpy(scaler.scale_).type(torch.float32),  # 1 / scale
            )

    @property
    def is_fitted(self) -> bool:
        return len(self._params) > 0

    def normalize_features_(self, data: T_CommonRoadData) -> T_CommonRoadData:

        for (storage, attr) in self._params.keys():
            mean, inv_scale = self._params[(storage, attr)]
            mean = mean.to(data.device)
            inv_scale = inv_scale.to(data.device)
            data[storage][attr] = (data[storage][attr] - mean) * inv_scale

        return data

    def unnormalize_features_(self, data: T_CommonRoadData) -> T_CommonRoadData:

        for (storage, attr) in self._params.keys():
            mean, inv_scale = self._params[(storage, attr)]
            mean = mean.to(data.device)
            inv_scale = inv_scale.to(data.device)
            data[storage][attr] = data[storage][attr] / inv_scale + mean

        return data

    def store_parameters(self, file: Path) -> None:
        torch.save({
            name: self._params[name] for name in self._params
        }, file)

    def load_parameters(self, file: Path) -> None:
        params = torch.load(file)
        self._params = params
        logger.info(f"Loaded parameters from '{file}'")


def normalize_features(
    dataset: CommonRoadDataset,
    max_fit_samples: Optional[int] = None,
    ignore_keys: Optional[Set[Tuple[str, str]]] = None
) -> FeatureNormalizer:

    feature_normalizer = FeatureNormalizer(ignore_keys=ignore_keys)

    # compute normalization parameters
    logger.info("Computing normalization parameters")
    if max_fit_samples is None:
        processed_files = dataset.processed_paths
    else:
        processed_files = dataset.processed_paths[:max_fit_samples]

    with ProgressReporter(total=len(processed_files) + 1, unit="sample") as progress:
        cpu = torch.device("cpu")
        for i, path in enumerate(processed_files):
            data = torch.load(path, map_location=cpu)
            if isinstance(data, tuple):
                data = data[-1]
            feature_normalizer.partial_fit(data)
            progress.update(i)
        progress.update(len(processed_files))

    def transform_normalize(scenario_index: int, sample_index: int,
                            data: T_CommonRoadData) -> Iterable[T_CommonRoadData]:
        yield feature_normalizer.normalize_features_(data)

    # normalize features
    logger.info("Normalizing features")
    feature_normalizer.prepare_parameters()
    dataset_transformation(dataset, transform=transform_normalize)

    # store normalization parameters
    normalization_params_file = get_normalization_params_file_path(
        Path(dataset.processed_dir)
    )
    feature_normalizer.store_parameters(file=normalization_params_file)

    return feature_normalizer
