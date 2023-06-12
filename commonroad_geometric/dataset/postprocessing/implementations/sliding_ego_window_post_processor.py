from typing import List, Optional

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle

class SlidingEgoWindowPostProcessor(BaseDataPostprocessor):
    """
    Inserts a virtual lanelet node at the current ego vehicle position. 
    """
    def __init__(
        self,
        lanelet_length: float
    ) -> None:
        """
        Args:
            lanelet_length (float): Length of ego-centered virtual lnaelet
        """
        self.lanelet_length = lanelet_length
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        raise NotImplementedError()
        