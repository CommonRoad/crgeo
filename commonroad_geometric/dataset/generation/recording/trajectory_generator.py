from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from pandas import DataFrame

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.dataset.generation.recording.sumo_recorder import SumoRecorder


class TrajectoryGenerator(AutoReprMixin):
    def __init__(
        self,
        sumo_recorder: SumoRecorder
    ) -> None:
        self._sumo_recorder = sumo_recorder

    def _copy_scenario(self) -> Scenario:
        initial_scenario = self._sumo_recorder.recorded_scenario
        generated_scenario = Scenario(
            dt=initial_scenario.dt,
            scenario_id=initial_scenario.scenario_id,
            author=initial_scenario.author,
            tags=initial_scenario.tags,
            affiliation=initial_scenario.affiliation,
            source=f"commonroad-geometric.TrajectoryRecorder and {initial_scenario.source}",
            location=initial_scenario.location
        )
        saved_lanelet_network = deepcopy(initial_scenario.lanelet_network)
        generated_scenario.add_objects(saved_lanelet_network)
        return generated_scenario

    def generate_scenario_with_trajectories(
        self,
        trajectory_ids: Iterable[int],
        ego_trajectory_id: Optional[int] = None,
        ego_obstacle_id: int = -1
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        """
        Args:
            trajectory_ids (Iterable[int]): the trajectory_ids to be included in this scenario
            ego_trajectory_id (Optional[int]): Trajectory which should be assigned to the ego obstacle.
            ego_obstacle_id (int): ID for ego dynamic obstacle in scenario.
        Returns:
            Scenario with trajectories of given trajectory_ids
        """
        generated_scenario = self._copy_scenario()

        for trajectory_id in trajectory_ids:
            trajectory = self._sumo_recorder.trajectory_id_to_trajectory[trajectory_id]
            dynamic_obstacle_id = generated_scenario.generate_object_id() if trajectory_id != ego_trajectory_id else ego_obstacle_id
            dynamic_obstacle = self._get_dynamic_obstacle_with_trajectory(trajectory_id=trajectory_id,
                                                                          trajectory=trajectory,
                                                                          dynamic_obstacle_id=dynamic_obstacle_id)
            generated_scenario.add_objects(dynamic_obstacle)

        ego_planning_problem_set = self._get_planning_problem_set(ego_trajectory_id)
        return generated_scenario, ego_planning_problem_set

    def generate_scenario_with_ego_trajectory(
        self,
        ego_trajectory_id: int,
        ego_obstacle_id: int = -1
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        """
        Args:
            ego_trajectory_id (int): trajectory_id for the trajectory
            ego_obstacle_id (int): ID to be used in scenario for saving ego trajectory

        Returns:
            Scenario with ego trajectory of given trajectory_id
        """
        generated_scenario, ego_planning_problem_set = self.generate_scenario_with_trajectories(trajectory_ids=[ego_trajectory_id],
                                                                                                ego_trajectory_id=ego_trajectory_id,
                                                                                                ego_obstacle_id=ego_obstacle_id)
        return generated_scenario, ego_planning_problem_set

    def _get_dynamic_obstacle_with_trajectory(
        self,
        trajectory_id: int,
        trajectory: List[State],
        dynamic_obstacle_id: int = -1
    ) -> DynamicObstacle:
        obstacle_in_initial_state = self._sumo_recorder.obstacle_in_initial_state(trajectory_id)
        assert obstacle_in_initial_state is not None

        initial_state = obstacle_in_initial_state.initial_state
        shape = obstacle_in_initial_state.obstacle_shape

        for time_step, state in enumerate(trajectory, start=initial_state.time_step):
            state.time_step = time_step

        trajectory = Trajectory(
            initial_time_step=initial_state.time_step,
            state_list=trajectory
        )

        trajectory_prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)
        return DynamicObstacle(
            obstacle_id=int(dynamic_obstacle_id),
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=shape,
            initial_state=initial_state,
            prediction=trajectory_prediction
        )

    def _get_planning_problem_set(self, trajectory_id: Optional[int]) -> Optional[PlanningProblemSet]:
        if trajectory_id is None:
            return None
        goal_region = self._get_goal_region(trajectory_id)
        obstacle_in_initial_state = self._sumo_recorder.obstacle_in_initial_state(trajectory_id)
        assert obstacle_in_initial_state is not None
        initial_state = obstacle_in_initial_state.initial_state
        # Requires yaw_rate, slip_angle to be set
        initial_state.yaw_rate = 0.0
        initial_state.slip_angle = 0.0

        planning_problem = PlanningProblem(planning_problem_id=42,
                                           initial_state=initial_state,
                                           goal_region=goal_region)
        planning_problem_set = PlanningProblemSet(planning_problem_list=[planning_problem])
        return planning_problem_set

    def _get_goal_region(self, trajectory_id: int) -> GoalRegion:
        # TODO Maybe set lanelets_of_goal_position attribute
        obstacle_in_initial_state = self._sumo_recorder.obstacle_in_initial_state(trajectory_id)
        obstacle_in_final_state = self._sumo_recorder.obstacle_in_final_state(trajectory_id)

        assert obstacle_in_initial_state is not None
        assert obstacle_in_final_state is not None

        initial_time_step = obstacle_in_initial_state.initial_state.time_step
        final_time_step = obstacle_in_final_state.prediction.final_time_step
        final_state = state_at_time(obstacle_in_final_state, final_time_step, assume_valid=True)
        goal_rectangle = Rectangle(length=obstacle_in_final_state.obstacle_shape.length,
                                   width=obstacle_in_final_state.obstacle_shape.width,
                                   center=final_state.position,
                                   orientation=final_state.orientation)
        return GoalRegion(state_list=[State(time_step=Interval(initial_time_step, final_time_step),
                                            velocity=Interval(0.0, 1.0),
                                            position=goal_rectangle)])

    def generate_scenario_with_trajectories_dataframe(
        self,
        trajectory_ids: Iterable[int],
        ego_trajectory_id: Optional[int] = None,
        ego_obstacle_id: int = -1
    ) -> Tuple[Scenario, Dict[int, DataFrame], Optional[PlanningProblemSet]]:
        """
        Args:
            trajectory_ids (Iterable[int]): the trajectory_ids to be included in this scenario
            ego_trajectory_id (Optional[int]): Trajectory which should be assigned to the ego obstacle.
            ego_obstacle_id (int): ID for ego dynamic obstacle in scenario.
        Returns:
            Scenario with full trajectory of ego_trajectory_id but without full trajectories of other trajectory_ids as these
            trajectories are saved separately in a dataframe
        """
        generated_scenario = self._copy_scenario()
        id_to_trajectory_dataframe: Dict[int, DataFrame] = {}

        for trajectory_id in trajectory_ids:
            trajectory = self._sumo_recorder.trajectory_id_to_trajectory[trajectory_id]
            # Save full trajectory for ego obstacle
            if trajectory_id == ego_trajectory_id:
                dynamic_obstacle = self._get_dynamic_obstacle_with_trajectory(trajectory_id=trajectory_id,
                                                                              trajectory=trajectory,
                                                                              dynamic_obstacle_id=ego_obstacle_id)
                generated_scenario.add_objects(dynamic_obstacle)
                continue
            # Save other trajectories in pandas dataframe in separate dictionary
            dynamic_obstacle_id = generated_scenario.generate_object_id()

            attributes = self._sumo_recorder.recorded_attributes
            trajectory_tuples = [tuple(getattr(state, attribute) for attribute in attributes) for state in trajectory]

            id_to_trajectory_dataframe[dynamic_obstacle_id] = DataFrame(trajectory_tuples, columns=self._sumo_recorder.recorded_attributes)

            # Only save initial and final state of trajectory in dynamic obstacle dummy of scenario
            dynamic_obstacle_dummy = self._get_dynamic_obstacle_stub_without_trajectory(trajectory_id=trajectory_id,
                                                                                        dynamic_obstacle_id=dynamic_obstacle_id)
            generated_scenario.add_objects(dynamic_obstacle_dummy)
        ego_planning_problem_set = self._get_planning_problem_set(ego_trajectory_id)
        return generated_scenario, id_to_trajectory_dataframe, ego_planning_problem_set

    def _get_dynamic_obstacle_stub_without_trajectory(
        self,
        trajectory_id: int,
        dynamic_obstacle_id: int = -1
    ) -> DynamicObstacle:
        obstacle_in_initial_state = self._sumo_recorder.obstacle_in_initial_state(trajectory_id)
        assert obstacle_in_initial_state is not None

        initial_recorded_state = obstacle_in_initial_state.initial_state
        shape = obstacle_in_initial_state.obstacle_shape

        obstacle_in_final_state = self._sumo_recorder.obstacle_in_final_state(trajectory_id)
        assert obstacle_in_final_state is not None

        final_time_step = obstacle_in_final_state.prediction.final_time_step
        final_recorded_state = state_at_time(obstacle_in_final_state, final_time_step, assume_valid=True)

        state_list = []
        for state in [initial_recorded_state, final_recorded_state]:
            kwargs = {attribute: state.__getattribute__(attribute) for attribute in self._sumo_recorder.recorded_attributes}
            state = State(time_step=state.time_step, **kwargs)
            state_list.append(state)

        trajectory = Trajectory(initial_time_step=initial_recorded_state.time_step,
                                state_list=state_list)
        trajectory_prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)

        return DynamicObstacle(obstacle_id=dynamic_obstacle_id,
                               obstacle_type=ObstacleType.CAR,
                               obstacle_shape=shape,
                               initial_state=state_list[0],
                               prediction=trajectory_prediction)
