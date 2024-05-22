from typing import List, Tuple, cast
from dataclasses import asdict
import numpy as np
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.io_extensions.mock_objects import create_dummy_lanelet, create_dummy_obstacle, create_dummy_scenario
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import BaseFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions


def _init_mock_setup(simulation_options: BaseSimulationOptions) -> Tuple[ScenarioSimulation, List[BaseFeatureParams]]:
    dummy_lanelet_1 = create_dummy_lanelet(
        lanelet_id=1000,
        successor=[1001],
        right_vertices=np.array([[0, 0], [1, 0], [2, 0], [3, .5], [4, 1], [5, 1], [6, 1], [7, 0], [8, 0]]),
        left_vertices=np.array([[0, 1], [1, 1], [2, 1], [3, 1.5], [4, 2], [5, 2], [6, 2], [7, 1], [8, 1]]),
        center_vertices=np.array([[0, .5], [1, .5], [2, .5], [3, 1], [4, 1.5], [5, 1.5], [6, 1.5], [7, .5], [8, .5]]),
    )
    dummy_lanelet_2 = create_dummy_lanelet(
        lanelet_id=1001,
        predecessor=[1000],
        right_vertices=np.array([[8, 0], [9, 0], [10, 0]]),
        left_vertices=np.array([[8, 1], [9, 1], [10, 1]]),
        center_vertices=np.array([[8, .5], [9, .5], [10, .5]]),
    )
    dummy_lanelet_network = LaneletNetwork.create_from_lanelet_list(
        [dummy_lanelet_1, dummy_lanelet_2], cleanup_ids=True)
    dummy_obstacles = []
    for i in range(2):
        dummy_obstacle = create_dummy_obstacle(
            obstacle_id=i,
            time_step=0,
            lanelet=dummy_lanelet_1,
            dist_along_lanelet=0.25 * i + 0.5,
        )
        dummy_obstacles.append(dummy_obstacle)
    dummy_states = [state_at_time(o, 0, assume_valid=True) for o in dummy_obstacles]
    dummy_scenario = create_dummy_scenario(
        lanelet_network=dummy_lanelet_network,
        dynamic_obstacles=dummy_obstacles
    )

    dummy_simulation_options = ScenarioSimulationOptions(
        collision_checking=False,
        backup_current_scenario=False,
        lanelet_assignment_order=simulation_options.lanelet_assignment_order,
        lanelet_graph_conversion_steps=simulation_options.lanelet_graph_conversion_steps,
        dt=simulation_options.dt,
        backup_initial_scenario=False,
        linear_lanelet_projection=True
    )
    dummy_simulation = ScenarioSimulation(
        initial_scenario=dummy_scenario,
        options=dummy_simulation_options
    )
    dummy_simulation.start()

    # This should just contain the dummy obstacle and nothing else
    assert len(dummy_scenario.dynamic_obstacles) == 2, "TrafficExtractor setup: Contains more than the dummy obstacles"
    for state in dummy_states:
        assert state is not None
        assert len(set(dummy_scenario.lanelet_network.find_lanelet_by_position([state.position])[
                   0])) > 0, "TrafficExtractor setup: Lanelet assignment for dummy state failed"

    class _Mock(object):
        pass
    dummy_params: List[BaseFeatureParams] = []
    for i in range(2):
        mock = _Mock()
        mock.dt = dummy_scenario.dt  # type: ignore
        mock.time_step = 0  # type: ignore
        mock.obstacle = dummy_obstacles[i]  # type: ignore
        mock.state = dummy_states[i]  # type: ignore
        mock.is_ego_vehicle = i == 0  # type: ignore
        mock.ego_state = dummy_states[0]  # type: ignore
        mock.ego_route = None  # type: ignore
        mock.distance = 0.0  # type: ignore
        mock.source_obstacle = dummy_obstacles[0]  # type: ignore
        mock.source_state = dummy_states[0]  # type: ignore
        mock.source_is_ego_vehicle = True  # type: ignore
        mock.target_obstacle = dummy_obstacles[1]  # type: ignore
        mock.target_state = dummy_states[1]  # type: ignore
        mock.target_is_ego_vehicle = False  # type: ignore
        mock.lanelet = dummy_lanelet_1  # type: ignore
        mock.source_lanelet = dummy_lanelet_1  # type: ignore
        mock.target_lanelet = dummy_lanelet_2  # type: ignore
        dummy_params.append(cast(BaseFeatureParams, mock))

    return dummy_simulation, dummy_params
