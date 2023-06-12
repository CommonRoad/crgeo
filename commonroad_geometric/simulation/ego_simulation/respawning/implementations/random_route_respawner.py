from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union

import networkx as nx
import numpy as np
from commonroad.scenario.trajectory import State

from commonroad_geometric.common.io_extensions.lanelet_network import lanelet_orientation_at_position, map_out_lanelets_to_intersections
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.types import Unlimited
from commonroad_geometric.simulation.ego_simulation.respawning.base_respawner import BaseRespawner, BaseRespawnerOptions, RespawnerSetupFailure, T_Respawn_Tuple

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

logger = logging.getLogger(__name__)


@dataclass
class RandomRouteRespawnerOptions(BaseRespawnerOptions):
    entry_offset_meters: float = 10.0
    exit_offset_meters: float = 2.5
    init_speed: float = 4.0

class RandomRouteRespawner(BaseRespawner):
    """
    Respawns ego vehicle at a randomly chosen entry point.
    """
    def __init__(
        self,
        options: Optional[RandomRouteRespawnerOptions] = None
    ) -> None:
        options = options or RandomRouteRespawnerOptions()
        self._options: RandomRouteRespawnerOptions = options
        super().__init__(options=options)

    def _get_respawn_tuple(self, ego_vehicle_simulation: EgoVehicleSimulation) -> T_Respawn_Tuple:
        routes = ego_vehicle_simulation.simulation.routes
        lanelet_network = ego_vehicle_simulation.simulation.lanelet_network

        entry_lanelet_id = self.rng.choice(list(routes.keys()))
        exit_lanelet_id = self.rng.choice(list(routes[entry_lanelet_id].keys()))
        exit_lanelet = lanelet_network.find_lanelet_by_id(exit_lanelet_id)

        entry_lanelet_polyline = ego_vehicle_simulation.simulation.get_lanelet_center_polyline(entry_lanelet_id)
        exit_lanelet_polyline = ego_vehicle_simulation.simulation.get_lanelet_center_polyline(exit_lanelet_id)
        
        entry_position = entry_lanelet_polyline(self._options.entry_offset_meters)
        entry_orientation = entry_lanelet_polyline.get_direction(self._options.entry_offset_meters)
        exit_position = exit_lanelet_polyline(exit_lanelet_polyline.length - self._options.entry_offset_meters)

        initial_state = State(
            position=entry_position,
            steering_angle=0.0,
            velocity=self._options.init_speed,
            orientation=entry_orientation,
            yaw_rate=0.0,
            slip_angle=0.0,
            time_step=ego_vehicle_simulation.current_time_step if ego_vehicle_simulation.current_time_step is not None else ego_vehicle_simulation.initial_time_step
        )
        return initial_state, exit_position, exit_lanelet
