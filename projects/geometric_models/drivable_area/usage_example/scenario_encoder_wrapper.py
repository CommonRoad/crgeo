import logging
from typing import List, Optional
import os
import numpy as np

from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions


# Get the directory of the current script
current_file_directory = os.path.dirname(__file__)


class ScenarioEncoderWrapper():
    """
    Wrapper for obtaining encodings from pretrained drivable area
    representation model.
    """

    def __init__(
        self,
        model_filepath: str
    ) -> None:

        self.model_filepath = model_filepath
        self.model = self._load_model()

        self.simulation: Optional[ScenarioSimulation] = None
        self.data_extractor: Optional[TrafficExtractor] = None

    def _load_model(self) -> BaseGeometric:
        model = BaseGeometric.load(
            self.model_filepath,
            device='cpu',
            retries=0,
            from_torch=False
        )
        model.eval()
        return model
    
    def load_scenario(self, scenario_path: str):
        self.simulation = ScenarioSimulation(scenario_path)
        self.simulation.start()

        extraction_options =TrafficExtractorOptions(
            edge_drawer=FullyConnectedEdgeDrawer(dist_threshold=100.0),
            feature_computers=TrafficFeatureComputerOptions(
                v=[
                    ft_veh_state,
                    VehicleLaneletConnectivityComputer(),
                ],
                v2v=[
                    ft_rel_state_ego,
                    LaneletDistanceFeatureComputer(
                        max_lanelet_distance=60.0,
                        max_lanelet_distance_placeholder=60.0 * 1.1,
                    ),
                    TimeToCollisionFeatureComputer(),
                ],
                l=[],
                l2l=[],
                v2l=[],
                l2v=[]
            ),
            assign_multiple_lanelets=True,
            postprocessors=[]
        )

        self.data_extractor = TrafficExtractor(self.simulation, options=extraction_options)



    def encode_current(self, time_step: int, reference_vehicle_id: int) -> np.ndarray:
        assert self.simulation is not None, "Please call load_scenario first"

        data = self.data_extractor.extract(time_step=time_step)
        encoding = self.model.encoder(data)

        ego_encoding = encoding[(data.v.id == reference_vehicle_id).squeeze(-1), :].squeeze(0)

        return ego_encoding.detach().numpy()
    

# usage example
if __name__ == '__main__':
    scenario_encoder = ScenarioEncoderWrapper(
        model_filepath=os.path.join(current_file_directory, "scenario_encoder_128_pretrained.pt")
    )
    
    scenario_encoder.load_scenario(os.path.join(current_file_directory, "DEU_LocationALower-11_2_T-1.xml"))

    reference_vehicle = scenario_encoder.simulation.current_scenario.dynamic_obstacles[0]

    encoding = scenario_encoder.encode_current(
        time_step=reference_vehicle.initial_state.time_step,
        reference_vehicle_id=reference_vehicle.obstacle_id
    )

    print(encoding) ## << [ 0.02373323  0.7078692  -0.24845353 ... ]