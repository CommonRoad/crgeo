from typing import Set, Tuple, Optional
import numpy as np
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

VELOCITY_THRESHOLD = 0.5


class TrafficJamCriterion(BaseTerminationCriterion):
    def __init__(
        self,
        velocity_threshold: float = 0.5,
        check_frequency: int = 10,
        check_all: bool = False,
        jam_distance_threshold: float = 10.0,
        behind_threshold: float = 10.0
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.check_frequency = check_frequency
        self.check_all = check_all
        self.jam_distance_threshold = jam_distance_threshold
        self.behind_threshold = behind_threshold
        self._call_count = -1
        super().__init__()

    def __call__(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        self._call_count += 1
        if self._call_count % self.check_frequency != 0:
            return False, None
        
        if len(ego_vehicle_simulation.current_lanelets) == 0:
            return False, None

        current_lanelet_id = ego_vehicle_simulation.current_lanelets[0].lanelet_id
        current_lanelet_polyline = ego_vehicle_simulation._simulation.get_lanelet_center_polyline(current_lanelet_id)
        obstacle_ids = ego_vehicle_simulation._simulation.get_obstacles_on_lanelet(
            current_lanelet_id,
            ignore_ids={ego_vehicle_simulation.ego_vehicle.obstacle_id}
        )
        if len(obstacle_ids) == 0:
            return False, None
        obstacle_states = [
            state_at_time(
                ego_vehicle_simulation._simulation.current_scenario._dynamic_obstacles[oid],
                ego_vehicle_simulation.current_time_step,
                assume_valid=True
            ) for oid in obstacle_ids
        ]
        obstacle_arclenghts = np.array([current_lanelet_polyline.get_projected_arclength(
            state.position,
            linear_projection=ego_vehicle_simulation.simulation.options.linear_lanelet_projection
        ) for state in obstacle_states])
        obstacle_speeds = np.array([state.velocity for state in obstacle_states])
        ego_arclength = current_lanelet_polyline.get_projected_arclength(
            ego_vehicle_simulation.ego_vehicle.state.position,
            linear_projection=ego_vehicle_simulation.simulation.options.linear_lanelet_projection
        )

        ego_distances = obstacle_arclenghts - ego_arclength
        stillstand_detections = obstacle_speeds[ego_distances > 0] < self.velocity_threshold
        closeness_detections = (ego_distances > 0) & (ego_distances < self.jam_distance_threshold)
        behind_detections = np.logical_and(ego_distances < 0, ego_distances > -self.behind_threshold)

        if self.check_all:
            is_traffic_jam = closeness_detections.any() and stillstand_detections.all() and behind_detections.any()
        else:
            is_traffic_jam = closeness_detections.any() and stillstand_detections.any() and behind_detections.any()

        return bool(is_traffic_jam), 'TrafficJam' if is_traffic_jam else None

    @property
    def reasons(self) -> Set[str]:
        return {'TrafficJam'}
