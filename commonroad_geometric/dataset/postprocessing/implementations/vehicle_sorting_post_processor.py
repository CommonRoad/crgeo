from dataclasses import dataclass
from torch import Tensor
from typing import Dict, Tuple
from typing import List, Optional
import torch

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


class VehicleSortingPostProcessor(BaseDataPostprocessor):
    """
    Facilitates ordering of vehicle nodes based on predefined metrics.
    """

    def __init__(
        self,
        sort_key: V_Feature,
        descending: bool = False
    ) -> None:
        """
        Args:
            lanelet_length (float): Length of ego-centered virtual lnaelet
        """
        self.sort_key = sort_key
        self.descending = descending
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        raise NotImplementedError()

        # start_index, end_index = feature_column_indices[self._sort_key]
        # indices = torch.argsort(x_vehicle[:, start_index:end_index], dim=0, descending=descending).squeeze(1)
        # return x_vehicle[indices], pos[indices], is_ego_mask[indices], vehicle_id[indices], indices
