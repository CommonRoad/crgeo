import sys, os; sys.path.insert(0, os.getcwd())

from typing import List

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.rendering.video_recording import render_scenario_movie
from commonroad_geometric.common.io_extensions.scenario import find_scenario_files, get_scenario_num_timesteps

NUM_SCENARIOS = 4
MIN_TIMESTEPS = 10
#MAX_TIMESTEPS = -5
MAX_TIMESTEPS = -5
VIDEO_OUTPUT_FILE = 'tutorials/output/scenario.gif'
INPUT_SCENARIO_MANY = 'data/osm_recordings'
INPUT_SCENARIO_ONE = 'data/osm_recordings/DEU_Munich-1_114_0_time_steps_1000_V1_0.xml'

if __name__ == '__main__':
    input_scenarios: List[Scenario] = []

    if NUM_SCENARIOS > 1:
        for scenario_file in sorted(find_scenario_files(INPUT_SCENARIO_MANY, skip_subvariants=True)):
            scenario, _ = CommonRoadFileReader(filename=scenario_file).open()
            n_time_steps = get_scenario_num_timesteps(scenario)
            if n_time_steps >= MIN_TIMESTEPS:
                input_scenarios.append(scenario)
                if len(input_scenarios) >= NUM_SCENARIOS:
                    break
    else:
        scenario, _ = CommonRoadFileReader(filename=INPUT_SCENARIO_ONE).open()
        input_scenarios.append(scenario)

    print(f"Rendering {len(input_scenarios)} scenarios: {[str(s.scenario_id) for s in input_scenarios]}")

    render_scenario_movie(
        input_scenarios,
        output_file=VIDEO_OUTPUT_FILE,
        max_timesteps=MAX_TIMESTEPS,
        save_pngs=True
    )

