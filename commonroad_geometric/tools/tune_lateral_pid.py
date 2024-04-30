from bayes_opt import BayesianOptimization
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.longitudinal_control_space import LongitudinalControlOptions, LongitudinalControlSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.simulation.ego_simulation.respawning.implementations.random_respawner import RandomRespawner, RandomRespawnerOptions
from commonroad_geometric.simulation.interfaces.static.unpopulated_simulation import UnpopulatedSimulation

TEST_REF_VALUES = [-0.4, 0.0, 0.4]


def run_scenario(
    ego_simulation: EgoVehicleSimulation,
    ref_value: float,
    render: bool,
    max_time_steps: Optional[float] = 200
) -> Tuple[float, float]:
    t = 0

    errors = []

    max_time_steps = max_time_steps if max_time_steps is not None else np.inf
    while t < max_time_steps:
        action = np.array([ref_value])
        for _ in ego_simulation.step(action):
            pass
        if render:
            ego_simulation.render(renderers=[])
        t += 1

        # if not ego_simulation.current_lanelet_ids:
        #     break
        if ego_simulation.check_if_completed_route():
            break

        path = ego_simulation.ego_route.extended_path_polyline
        pos = ego_simulation.ego_vehicle.state.position
        # arclength = path.get_projected_arclength(pos)
        error = abs(path.get_lateral_distance(pos))
        errors.append(error)

    ego_simulation.reset(respawn_until_success=True)

    errors = np.array(errors)
    mean_error = errors.mean()
    max_error = errors.max()

    return mean_error, max_error


def evaluate(
    respawner: RandomRespawner,
    scenarios: List[ScenarioBundle],
    simulation_dict: Dict[str, BaseSimulation],
    renderer: Optional[TrafficSceneRenderer],
    look_ahead_distance: float,
    k_e: float,
    k_v: float,
    k_dy: float,
    k_ds: float,
    k_ss: float
) -> float:
    mean_errors = []
    max_errors = []

    for ref_value in TEST_REF_VALUES:

        for scenario in scenarios:
            scenario_key = str(scenario.scenario_id)
            # print(f"{scenario_key}")

            if scenario_key in simulation_dict:
                simulation = simulation_dict[scenario_key]
                simulation.despawn_ego_vehicle(None)
                simulation.close()
            else:
                simulation = UnpopulatedSimulation(scenario)
                simulation_dict[scenario_key] = simulation

            ego_simulation = EgoVehicleSimulation(
                simulation=simulation,
                control_space=LongitudinalControlSpace(
                    LongitudinalControlOptions(
                        look_ahead_distance=look_ahead_distance,
                        k_e=k_e,
                        k_v=k_v,
                        k_dy=k_dy,
                        k_ds=k_ds,
                        k_ss=k_ss,
                        pid_control=True,
                        max_velocity=8.0
                    )
                ),
                respawner=respawner,
                renderers=[renderer]
            )
            ego_simulation.respawner.rng.seed(0)
            ego_simulation.start(respawn_until_success=True)

            try:
                mean_error, max_error = run_scenario(ego_simulation, ref_value, render=renderer is not None)
                mean_errors.append(mean_error)
                max_errors.append(max_error)
                # print(f"{scenario_key}: {ref_value=}, {mean_error=:.3f}, {max_error=:.3f}")
            except Exception as e:
                print(e)
            ego_simulation.close()

    mean_errors = np.array(mean_errors)
    max_errors = np.array(max_errors)

    avg_mean_error = mean_errors.mean()
    avg_max_error = max_errors.mean()

    score = -(avg_max_error * 0.2 + avg_mean_error)
    # print(f"{avg_mean_error=:.3f}, {avg_max_error=:.3f}, {score=:.3f}")
    return score


def main():
    scenario_iterator = ScenarioIterator('data/osm_recordings', max_scenarios=3)
    scenarios = [s.preprocessed_scenario for s in scenario_iterator]
    simulation_dict = {}
    renderer = None
    # renderer = TrafficSceneRenderer()

    respawner = RandomRespawner(
        RandomRespawnerOptions(
            min_goal_distance=None,
            max_goal_distance=None,
            min_curvature=0.4,
            init_speed=2.0,
            only_intersections=False,
            route_length=2,
            use_cached=True,
            min_remaining_distance=None,
            max_respawn_attempts=100,
            random_init_arclength=False,
            random_goal_arclength=False
        )
    )

    eval_fn = partial(
        evaluate,
        respawner=respawner,
        scenarios=scenarios,
        simulation_dict=simulation_dict,
        renderer=renderer)

    # Bounded region of parameter space
    pbounds = dict(
        look_ahead_distance=(0.0, 2.5),
        k_e=(0.0, 10.0),
        k_v=(3.0, 20.0),
        k_dy=(0.1, 3.0),
        k_ds=(0.1, 3.0),
        k_ss=(0.0, 0.2),
        # gain=(0.7, 1.3)
    )

    optimizer = BayesianOptimization(
        f=eval_fn,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=100,
        n_iter=10000,
    )
    print(optimizer.max)

    print('done')


if __name__ == '__main__':
    with warnings.catch_warnings(record=False) as w:
        warnings.simplefilter("ignore")
        # Trigger a warning.
        main()
