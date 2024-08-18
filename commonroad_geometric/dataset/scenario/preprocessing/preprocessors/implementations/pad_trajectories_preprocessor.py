from __future__ import annotations
import copy
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor


class PadTrajectoriesPreprocessor(ScenarioPreprocessor):

    def __init__(self) -> None:
        super(PadTrajectoriesPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        max_time_step = 0

        # Find the greatest time_step of all the vehicle trajectories in the scenario
        for obstacle in scenario_bundle.preprocessed_scenario.dynamic_obstacles:
            if obstacle.prediction.trajectory.state_list:
                max_time_step = max(max_time_step, obstacle.prediction.trajectory.state_list[-1].time_step)

        for obstacle in scenario_bundle.preprocessed_scenario.dynamic_obstacles:
            trajectory = obstacle.prediction.trajectory.state_list

            if not trajectory:
                continue
            obstacle.prediction.trajectory._initial_time_step = 0
            
            # Get the first and last state
            first_state = trajectory[0]
            last_state = trajectory[-1]

            # Pad the trajectory backwards with the first state until t=0
            first_time_step = first_state.time_step
            if first_time_step > 0:
                padded_states = []
                for t in range(first_time_step):
                    new_state = copy.deepcopy(first_state)
                    new_state.time_step = t
                    padded_states.append(new_state)
                trajectory = padded_states + trajectory

            # Pad the trajectory forwards with the last state until the max_time_step
            last_time_step = last_state.time_step
            if last_time_step < max_time_step:
                padded_states = []
                for t in range(last_time_step + 1, max_time_step + 1):
                    new_state = copy.deepcopy(last_state)
                    new_state.time_step = t
                    padded_states.append(new_state)
                trajectory = trajectory + padded_states

            # Update the trajectory with the padded states
            obstacle.prediction.trajectory._state_list = trajectory

        return [scenario_bundle]
