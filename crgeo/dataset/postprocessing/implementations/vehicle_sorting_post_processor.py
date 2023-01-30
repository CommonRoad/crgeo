from typing import List, Optional

from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from crgeo.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from crgeo.simulation.base_simulation import BaseSimulation
from crgeo.simulation.ego_simulation.ego_vehicle import EgoVehicle


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
