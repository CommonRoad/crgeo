import sys; import os; sys.path.insert(0, os.getcwd())

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from commonroad_geometric.common.torch_utils.helpers import flatten_data
from commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.feature_computers import BaseFeatureComputer, DefaultFeatureComputers, FeatureDict, VFeatureParams
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from commonroad_geometric.simulation.simulation_factory import SimulationFactory


# Tutorials demonstrate how CommonRoad-Geometric should be used.
# Do not modify this for your own purposes. Create a tool or project instead.
class TutorialModule(nn.Module):
    def __init__(self) -> None:
        super(TutorialModule, self).__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x)  # type: ignore


class TutorialFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    """
    Dummy feature computer that showcases the flexibility of the feature computer framework.
    """

    def __init__(
        self,
        pretrained_model: TutorialModule,
        velocity_offset: float = 2.0,
    ) -> None:
        # We can provide a pre-trained model and access it for feature computation.
        self._pretrained_model = pretrained_model

        # We can allow the user to customize parameters for the output features.
        self._velocity_offset = velocity_offset
        self._vehicle_call_count: dict[int, int]
        super(TutorialFeatureComputer, self).__init__()

    @property
    def name(self) -> str:
        return type(self).__name__

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        """
        The __call__ method is being called for each obstacle at each time-step, and returns the computed features.

        Args:
            params (VFeatureParams):
                Struct containing references to the vehicle and its state.
            simulation:
                BaseSimulation that keeps track of the traffic scene, allowing e.g. lanelet lookup for obstacles.

        Returns:
            FeatureDict:
                Dictionary mapping from feature name to feature values of valid types,
                i.e. either floats, integers, booleans or 1-d PyTorch tensors.
        """

        # The BaseFeatureComputer framework supports statefulness.
        # Here we are simply keeping track of the execution counts for each vehicle.
        self._vehicle_call_count[params.obstacle.obstacle_id] += 1

        # We use the simulation instance to obtain further information about the vehicle.
        vehicle_lanelet = simulation.get_obstacle_lanelet(params.obstacle)
        assert vehicle_lanelet is not None
        has_adjacent_lane_left = vehicle_lanelet.adj_left_same_direction == True

        # We can use our pre-trained model to compute learned state representations.
        x_in = torch.tensor([params.state.orientation, params.state.velocity], dtype=torch.float32)
        prediction = self._pretrained_model(x_in).item()

        # We can compute new features based on previously computed ones, potentially saving computation time.
        velocity_8 = self.ComputedFeaturesCache['velocity_4'] ** 2

        # We define the features to be included in the resulting Data instance.
        features = {
            'prediction': prediction,
            'call_count': self._vehicle_call_count[params.obstacle.obstacle_id],
            'velocity_8': velocity_8 + self._velocity_offset,
            'has_adjacent_lane_left': has_adjacent_lane_left,
            # type: ignore
            'shape_tensor': torch.tensor([params.obstacle.obstacle_shape.width, params.obstacle.obstacle_shape.length], dtype=torch.float32)
        }

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        # The reset method is called at the beginning of a new scenario.
        self._vehicle_call_count = defaultdict(int)


if __name__ == '__main__':
    # We define the feature computers for vehicle nodes.
    # They will be executed in the given order.
    custom_vehicle_node_feature_computers = [
        # Lambda functions allow simple implementations for trivial features
        lambda params: dict(velocity_2=params.state.velocity ** 2),

        # Nested feature computations done via accessing the cached values.
        lambda params: dict(velocity_4=BaseFeatureComputer.ComputedFeaturesCache['velocity_2'] ** 2),

        # Our custom computer with more involved internal operations than a lambda function would allow.
        TutorialFeatureComputer(
            pretrained_model=TutorialModule(),
            velocity_offset=5.0
        ),
    ]

    # Creating a collector with our custom vehicle_node_feature_computers
    collector = DatasetCollector(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
                feature_computers=TrafficFeatureComputerOptions(
                    v=DefaultFeatureComputers.v() + custom_vehicle_node_feature_computers
                ),
            )
        ),
        simulation_factory=SimulationFactory(ScenarioSimulationOptions()),
        progress=False
    )

    # Running the collector
    for scenario_bundle in ScenarioIterator(directory=Path('data/highd-sample')):
        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            max_samples=5,
        ):
            print(f"{time_step=}:\n{data=}")
            # You can look at this with a debugger
            flattened_data = flatten_data(data, 1000, validate=True)
