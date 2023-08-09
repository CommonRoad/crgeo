import inspect
import os
import sys

sys.path.insert(0, os.getcwd())

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.debugging.profiling import profile
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_game import EgoVehicleGame
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.implementations import ConstantRateSpawner
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation

INTERACTIVE = True
VIDEO_OUTPUT_DIR = 'tutorials/output/ego_game'
VIDEO_FREQUENCY = 500
VIDEO_START_RECORDING_AT = 100
INPUT_SCENARIO = 'data/osm_recordings/DEU_Munich-1_114_0_time_steps_1000_V1_0.xml'
PRESIMULATION_STEPS = 0  # 'auto'


def play(
    scenario: Scenario,
    planning_problem_set: PlanningProblemSet,
    simulation: BaseSimulation
) -> None:
    """Lets you play the ego vehicle using the keyboard"""

    game = EgoVehicleGame(
        scenario=scenario,
        simulation=simulation,
        planning_problem_set=planning_problem_set,
        traffic_extractor_options=TrafficExtractorOptions(
            edge_drawer=VoronoiEdgeDrawer(dist_threshold=25.0),
        ),
        traffic_spawner=ConstantRateSpawner(p_spawn=0.01)
    )

    i = 0
    video_count = 0
    game.start()
    while game.running:
        game.step()
        i += 1
        if i >= VIDEO_START_RECORDING_AT:
            if i == VIDEO_START_RECORDING_AT:
                game.clear_frames()
                print("Started recording")
            if (i + VIDEO_START_RECORDING_AT) % VIDEO_FREQUENCY == 0:
                output_file = os.path.join(VIDEO_OUTPUT_DIR, f'video_{video_count}.gif')
                game.save_video(output_file=output_file, save_pngs=True)
                video_count += 1


if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(filename=INPUT_SCENARIO).open()
    simulation: BaseSimulation
    if INTERACTIVE:
        simulation = SumoSimulation(
            initial_scenario=scenario,
            options=SumoSimulationOptions(
                presimulation_steps=PRESIMULATION_STEPS
            )
        )
    else:
        simulation = ScenarioSimulation(
            initial_scenario=INPUT_SCENARIO,
        )

    profile(
        func=play,
        kwargs=dict(
            scenario=scenario,
            planning_problem_set=planning_problem_set,
            simulation=simulation
        )
    )
