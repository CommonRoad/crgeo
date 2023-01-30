import os
import sys

sys.path.insert(0, os.getcwd())

from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn

from crgeo.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.feature_computers import VFeatureParams
from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer, FeatureDict
from crgeo.dataset.extraction.traffic.feature_computers.implementations.defaults import DefaultFeatureComputers
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from crgeo.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from crgeo.dataset.iteration import ScenarioIterator
from crgeo.simulation.base_simulation import BaseSimulation

SCENARIO_DIR = 'data/highway_test'
MAX_SAMPLES = 200


class TutorialPytorchModule(nn.Module):
    def __init__(self) -> None:
        super(TutorialPytorchModule, self).__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) # type: ignore


class TutorialFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    """
    Dummy feature computer that showcases the flexibilty of the feature computer framework.
    """

    def __init__(
        self,
        pytorch_module: TutorialPytorchModule,
        velocity_offset: float = 2.0,
    ) -> None:
        # We can provide a pre-trained model and access it for feature computation.
        self._pytorch_module = pytorch_module

        # We can allow the user to customize parameters for the outputted features.
        self._velocity_offset = velocity_offset
        super(TutorialFeatureComputer, self).__init__()

    @property
    def name(self) -> str:
        return "TutorialFeatureComputer"

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
        prediction = self._pytorch_module(x_in).item()

        # We can compute new features based on previously computed ones, potentially saving computation time.
        velocity_8 = self.ComputedFeaturesCache['velocity_4'] ** 2

        # We define the features to be included in the resulting Data instance.
        features = {
            'prediction': prediction,
            'call_count': self._vehicle_call_count[params.obstacle.obstacle_id],
            'velocity_8': velocity_8 + self._velocity_offset,
            'has_adjacent_lane_left': has_adjacent_lane_left,
            'shape_tensor': torch.tensor([params.obstacle.obstacle_shape.width, params.obstacle.obstacle_shape.length], dtype=torch.float32) # type: ignore
        }

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        # The reset method is called at the beginning of a new scenario.
        self._vehicle_call_count: Dict[int, int] = defaultdict(int)


if __name__ == '__main__':
    from crgeo.common.torch_utils.helpers import flatten_data

    # We define the feature computers for vehicle nodes.
    # They will be executed in the given order.
    custom_vehicle_node_feature_computers = [
        # Lambda functions allow simple implementations for trivial features
        lambda params: dict(velocity_2=params.state.velocity ** 2),

        # Nested feature computations done via accessing the cached values.
        lambda params: dict(velocity_4=BaseFeatureComputer.ComputedFeaturesCache['velocity_2'] ** 2),

        # Our custom computer with more involved internal operations than a lambda function would allow.
        TutorialFeatureComputer(
            pytorch_module=TutorialPytorchModule(),
            velocity_offset=5.0
        ),
    ]

    # Creating a collector with our custom vehicle_node_feature_computers
    collector = ScenarioDatasetCollector(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
                feature_computers=TrafficFeatureComputerOptions(
                    v=DefaultFeatureComputers.v() + custom_vehicle_node_feature_computers
                ),
            )
        )
    )

    # Running the collector
    for scenario_bundle in ScenarioIterator(SCENARIO_DIR):
        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario, 
            max_samples=MAX_SAMPLES
        ):
            print(data)
            
            
            
            flattened_data = flatten_data(data, 1000, validate=True)

