import os
import sys

sys.path.insert(0, os.getcwd())

import pyglet
pyglet.options['headless'] = True

import shutil
import unittest
from typing import Iterable

import torch
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
import torch_geometric.loader
from tutorials.render_scenario_from_graph import pre_transform

from crgeo.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.commonroad_dataset import CommonRoadDataset
from crgeo.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph
from crgeo.dataset.extraction.road_network.implementations import IntersectionGraph, LaneletEndpointGraph, LaneletGraph
from crgeo.dataset.extraction.road_network.road_network_extractor import RoadNetworkExtractorOptions
from crgeo.dataset.extraction.road_network.road_network_extractor_factory import RoadNetworkExtractorFactory
from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions, TrafficFeatureComputerOptions
from crgeo.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from crgeo.dataset.generation.recording.trajectory_recorder import TrajectoryRecorder
from crgeo.dataset.iteration import ScenarioIterator, TimeStepIterator
from crgeo.dataset.preprocessing.implementations import DepopulateScenarioPreprocessor
from crgeo.rendering.plugins import RenderLaneletNetworkPlugin, RenderObstaclesPlugin
from crgeo.rendering.types import T_Frame
from crgeo.simulation.base_simulation import BaseSimulation
from crgeo.simulation.ego_simulation.ego_vehicle_game import EgoVehicleGame
from crgeo.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from crgeo.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from crgeo.simulation.interfaces.interactive.traffic_spawning.implementations import ConstantRateSpawner
from crgeo.simulation.interfaces.static.scenario_simulation import ScenarioSimulation
from tutorials.render_scenario_from_graph import pre_transform

INPUT_SCENARIO = 'data/other/ARG_Carcarana-1_7_T-1.xml'


class TestTutorials(unittest.TestCase):
    """
    _summary_

    This script servers as a pre-check of basic functionalities.
    It runs tutorials for restricted time steps in limited scenarios.

    If the test failed with your new feature, you can identify the corresponding tutorials that raised the error.
    If you see RuntimeError: Max respawn attempts reached (50), you can clear the terminal and run the test again.
    If you changed some interfaces, please fix them in both the tutorials and this test.

    """

    def setUp(self) -> None:
        self.scenario, self.planning_problem_set = CommonRoadFileReader(filename=INPUT_SCENARIO).open()

    # modified from tutorial play_ego_vehicle
    def test_play_ego_vehicle_interactive(self) -> None:

        simulation = SumoSimulation(
            initial_scenario=self.scenario,
            options=SumoSimulationOptions(
                presimulation_steps=0
            )
        )

        game = EgoVehicleGame(
            scenario=self.scenario,
            simulation=simulation,
            traffic_extractor_options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=25.0),
            ),
            traffic_spawner=ConstantRateSpawner(p_spawn=0.05)
        )
        game.start()
        game.step()
        simulation.close()
        self.assertIsNotNone(game._video_frames[0])
        self.assertIsInstance(game._ego_sim, EgoVehicleSimulation)

    # modified from tutorial play_ego_vehicle
    def test_play_ego_vehicle_noninteractive(self) -> None:
        DEPOPULATE = True
        DEPOPULATE_VALUE = 5
        simulation: BaseSimulation

        simulation = ScenarioSimulation(
            initial_scenario=INPUT_SCENARIO,
        )

        if DEPOPULATE:
            depopulator = DepopulateScenarioPreprocessor(DEPOPULATE_VALUE)
            depopulator(simulation.current_scenario)

        game = EgoVehicleGame(
            scenario=self.scenario,
            simulation=simulation,
            traffic_extractor_options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=25.0),
            ),
            traffic_spawner=ConstantRateSpawner(p_spawn=0.05)
        )
        game.start()
        game.step()
        game.close()

        self.assertIsNotNone(game._video_frames[0], T_Frame)
        self.assertIsInstance(game._ego_sim, EgoVehicleSimulation)

    # modified from tutorial enjoy_scenario_simulation
    def test_scenario_simulation(self) -> None:
        simulation = ScenarioSimulation(
        initial_scenario=INPUT_SCENARIO
    )
        DEPOPULATE = True
        DEPOPULATE_VALUE =  5

        if DEPOPULATE:
            depopulator = DepopulateScenarioPreprocessor(DEPOPULATE_VALUE)
            depopulator(simulation.current_scenario, None)

        extractor = TrafficExtractor(
            simulation=simulation,
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(),
            )
        )
        simulation.start()
        for time_step, scenario in simulation:
            data = extractor.extract(TrafficExtractionParams(
                index=time_step,
            ))
            if time_step > 10: 
                break
        simulation.reset()
        assert isinstance(simulation.final_time_step, int)
        for time_step, scenario in simulation(from_time_step=simulation.initial_time_step, to_time_step=min(simulation.initial_time_step + 10, simulation.final_time_step)):
            pass
        simulation.close()

    # modified from tutorial enjoy_sumo_simulation
    def test_sumo_simulation(self) -> None:
        simulation = SumoSimulation(
            initial_scenario=INPUT_SCENARIO,
            options=SumoSimulationOptions(
                presimulation_steps=10,
                render_plugins=[
                    RenderLaneletNetworkPlugin(),
                    RenderObstaclesPlugin()
                ]
            )
        )

        simulation.start()
        for time_step, scenario in simulation(from_time_step=0, to_time_step=20):
            pass
        simulation.close()

    # modified from tutorial extract_and_plot_road_network_graph
    def test_extract_and_plot_road_network_graph(self) -> None:
        output_dirs = ['tutorials/output/road_network_graphs/endpoint', 'tutorials/output/road_network_graphs/lanelet', 'tutorials/output/road_network_graphs/intersection']
        configurations = [
            (LaneletEndpointGraph, True, False, False, False, False, dict(waypoint_density=15), dict(failed=0, successful=0), output_dirs[0]),
            (LaneletGraph, True, False, False, True, True, {}, dict(failed=0, successful=0), output_dirs[1]),
            (IntersectionGraph, True, False, True, False, True, {}, dict(failed=0, successful=0), output_dirs[2]),
        ]

        for cls, show_waypoints, show_edge_weights, show_edge_angles, show_edge_labels, show_node_labels, conversion_options, conversion_summary, output_dir in configurations:
            print(f"Converting to {cls.__name__}")
            graph: BaseRoadNetworkGraph = cls.from_scenario_file(
                INPUT_SCENARIO, **conversion_options)
            self.assertIsNotNone(graph)

    # modified from tutorial extract_and_render_traffic_graph
    def test_extract_and_render_traffic_graph(self) -> None:
        traffic_extractor = TrafficExtractor(
            simulation=ScenarioSimulation(initial_scenario=INPUT_SCENARIO),
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer()
            )
        )
        traffic_extractor.simulation.start()
        self.assertIsNotNone(traffic_extractor)
        # for data in traffic_extractor:
        #     print(data)

    # modified from tutorial record_trajectories
    def test_record_trajectories(self) -> None:
        options = SumoSimulationOptions(presimulation_steps=0)
        simulation = SumoSimulation(
            initial_scenario=INPUT_SCENARIO,
            options=options
        )
        trajectory_recorder = TrajectoryRecorder(sumo_simulation=simulation)
        trajectory_recorder.start_simulation()
        trajectory_id_to_trajectory = trajectory_recorder.record_trajectories_for_time_steps(
            time_steps=50,
            min_trajectory_length=10
        )
        self.assertIsNotNone(trajectory_id_to_trajectory)
        trajectory_recorder.close_simulation()

    # modified from tutorial iterate over scenarios
    def test_scenario_iterator(self) -> None:
        SCENARIO_DIR = 'data/other'
        SCENARIO_PREPROCESSORS = [
            (DepopulateScenarioPreprocessor(5), 3),
        ]
        iterator = ScenarioIterator(
            SCENARIO_DIR,
            preprocessors=SCENARIO_PREPROCESSORS,
            load_scenario_pickles=False,
            save_scenario_pickles=False,
            verbose=1
        )
        for scenario_bundle in iterator:
            self.assertIsNotNone(scenario_bundle)

    # modified from tutorial generate scenario from graph
    def test_generate_scenario_from_graph(self) -> None:
        DATASET_DIR = 'tutorials/output/dataset/highd'
        SCENARIO_DIR = 'data/highway_test'
        NUM_WORKERS = 1
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
        dataset = CommonRoadDataset(
            raw_dir=SCENARIO_DIR,
            processed_dir=DATASET_DIR,
            pre_transform=pre_transform,
            pre_transform_workers=NUM_WORKERS,
        )
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
        for index, batch in enumerate(loader):
            scenario, _ = CommonRoadFileReader([x for x in dataset.raw_paths if batch[1].scenario_id[0] in x][0]).open()
            # Load scenario
            timestep_iterator = TimeStepIterator(scenario, loop=True)

    # modified from tutorial collect_traffic_data_with_custom_features
    def test_collect_traffic_data_with_custom_features(self) -> None:
        from crgeo.common.torch_utils.helpers import flatten_data
        INPUT_SCENARIO = 'data/highd-test/DEU_LocationALower-11_1_T-1.xml'
        N_SAMPLES = 40

        class TestFeatureComputer(BaseFeatureComputer[VFeatureParams]):

            def __init__(self) -> None:
                self._call_count = 0
                self._reset_count = 0
                super(TestFeatureComputer, self).__init__()

            @property
            def name(self) -> str:
                return "TestFeatureComputer"

            def __call__(
                self,
                params: VFeatureParams,
                simulation: BaseSimulation,
            ) -> FeatureDict:
                features: FeatureDict = {
                    'reset_count': self._reset_count,
                    'call_count': self._call_count,
                    'time_step': params.time_step
                }
                self._call_count += 1

                return features

            def _reset(self, simulation: BaseSimulation) -> None:
                # The reset method is called at the beginning of a new scenario.
                self._call_count = 0
                self._reset_count += 1

        # We define the feature computers for vehicle nodes.
        # They will be executed in the given order.
        custom_vehicle_node_feature_computers = [
            # Lambda functions allow simple implementations for trivial features
            lambda params: dict(velocity_2=params.state.velocity ** 2),

            # Nested feature computations done via accessing the cached values.
            lambda params: dict(velocity_4=BaseFeatureComputer.ComputedFeaturesCache['velocity_2'] ** 2),

            # Our custom computer with more involved internal operations than a lambda function would allow.
            TestFeatureComputer(),
        ]

        # Creating a collector with our custom vehicle_node_feature_computers
        collector = ScenarioDatasetCollector(
            extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
                    feature_computers=TrafficFeatureComputerOptions(
                        v=custom_vehicle_node_feature_computers
                    )
                )
            )
        )

        scenario_bundles = [scenario_bundle for scenario_bundle in ScenarioIterator(INPUT_SCENARIO)]
        self.assertEqual(len(scenario_bundles), 1)

        dataset = [sample for time_step, sample in collector.collect(scenario=scenario_bundles[0].preprocessed_scenario, max_samples=N_SAMPLES)]
        repr(dataset[0])
        self.assertEqual(len(dataset), N_SAMPLES)

        flattened_dataset = []
        for data_idx, data in enumerate(dataset):
            flattened_data = flatten_data(data, 1000, validate=False)
            self.assertTrue((data.vehicle.time_step == data_idx).all())
            flattened_dataset.append(flattened_data)

        # creating batched data
        batched_data = {}
        for key in flattened_dataset[0].keys():
            all_values = [flattened_data[key] for flattened_data in flattened_dataset]
            batched_values = torch.stack(all_values, dim=0)
            batched_data[key] = batched_values

        # reconstructing batched data
        reconstructed_batched_data = CommonRoadData.reconstruct(batched_data)

        batch_size = reconstructed_batched_data.vehicle.batch.max().item() + 1
        self.assertEqual(batch_size, N_SAMPLES)

        from torch_geometric.utils import subgraph

        for batch_idx in range(batch_size):

            # vehicle
            batch_mask_v = reconstructed_batched_data.vehicle.batch == batch_idx
            sub_edge_index_v, sub_edge_attr_v, sub_edge_mask_v = subgraph(
                batch_mask_v,
                reconstructed_batched_data.vehicle_to_vehicle.edge_index,
                reconstructed_batched_data.vehicle_to_vehicle.edge_attr,
                relabel_nodes=True,
                return_edge_mask=True
            )
            edge_index_v = dataset[batch_idx].vehicle_to_vehicle.edge_index
            edge_attr_v = dataset[batch_idx].vehicle_to_vehicle.edge_attr
            self.assertTrue((edge_index_v == sub_edge_index_v).all())
            self.assertTrue((edge_attr_v == sub_edge_attr_v).all())

            # lanelet
            batch_mask_l = reconstructed_batched_data.lanelet.batch == batch_idx
            sub_edge_index_l, sub_edge_attr_l, sub_edge_mask_l = subgraph(
                batch_mask_l,
                reconstructed_batched_data.lanelet_to_lanelet.edge_index,
                reconstructed_batched_data.lanelet_to_lanelet.weight,
                relabel_nodes=True,
                return_edge_mask=True
            )
            edge_index_l = dataset[batch_idx].lanelet_to_lanelet.edge_index
            edge_attr_l = dataset[batch_idx].lanelet_to_lanelet.weight
            self.assertTrue((edge_index_l == sub_edge_index_l).all())
            self.assertTrue((edge_attr_l == sub_edge_attr_l).all())

            # vehicle-to-lanelet edges
            batch_indices_v = torch.where(batch_mask_v)[0]
            edge_index_v2l = dataset[batch_idx].vehicle_to_lanelet.edge_index
            batched_edge_index_v2l = reconstructed_batched_data.vehicle_to_lanelet.edge_index
            sub_edge_mask_v2l = torch.eq(batched_edge_index_v2l[0, :].unsqueeze(1), batch_indices_v.unsqueeze(0).repeat(
                batched_edge_index_v2l.shape[1], 1)
            ).sum(dim=1).bool()
            sub_edge_index_v2l = batched_edge_index_v2l[:, sub_edge_mask_v2l]
            sub_edge_index_v2l_relabeled = sub_edge_index_v2l - sub_edge_index_v2l.min(dim=1).values[:, None]
            self.assertTrue((edge_index_v2l == sub_edge_index_v2l_relabeled).all())

        return 1
    
    #modified from tutorial collect_traffic_data_sumo
    def test_collect_traffic_data_sumo(self) -> None:
        PRESIMULATION_STEPS = 0
        simulation = SumoSimulation(
            initial_scenario=INPUT_SCENARIO,
            options=SumoSimulationOptions(
                presimulation_steps=PRESIMULATION_STEPS
            )
        )

        traffic_extractor = TrafficExtractor(
            simulation=simulation,
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=25),
            )
        )

        samples = []

        traffic_extractor.simulation.start()

        for time_step, scenario in traffic_extractor.simulation(to_time_step=50):
            data = traffic_extractor.extract(TrafficExtractionParams(
                index=time_step,
            ))
            samples.append(data)

        traffic_extractor.simulation.close()

        data_loader = torch_geometric.loader.DataLoader(
            samples,
            batch_size=10,
            shuffle=True
        )
        self.assertIsNotNone(data_loader)

    #modified from tutorial collect_road_network_dataset
    def test_collect_road_network_dataset(self) -> None:
        shutil.rmtree("tutorials/output/graphdata", ignore_errors=True)

        # check out https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html and
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset for more details
        dataset = CommonRoadDataset(
            raw_dir="data/other",
            processed_dir="tutorials/output/graphdata",
            # the pre_transform function transforms a CommonRoad scenario to any number of dataset samples
            pre_transform=pre_transform_roadnetwork,
            # you can optionally parallelize the pre_transform step by using multiple worker processes
            pre_transform_workers=4,
        )
        self.assertIsNotNone(dataset)


def pre_transform_roadnetwork(scenario: Scenario, planning_problem_set: PlanningProblemSet) -> Iterable[CommonRoadData]:
    extractor_factory = RoadNetworkExtractorFactory(RoadNetworkExtractorOptions(
        graph_cls=LaneletEndpointGraph,
        max_size=3,
        depth=2,
        include_radius=500.0,
        exclude_leaf_nodes=True,
        plot=False
    ))

    collector = ScenarioDatasetCollector(
        extractor_factory=extractor_factory
    )
    for sample in collector.collect(scenario, max_samples=10, progress=False):
        yield sample


if __name__ == "__main__":
    unittest.main()
