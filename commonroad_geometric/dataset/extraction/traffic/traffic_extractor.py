from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from torch_geometric.utils import subgraph

from commonroad_geometric.common.io_extensions.obstacle import map_obstacle_type, state_at_time
from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.common.utils.string import lchop
from commonroad_geometric.common.torch_utils.helpers import keys
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.base_extractor import (BaseExtractor,
                                                                    BaseExtractionParams,
                                                                    BaseExtractorOptions)
from commonroad_geometric.dataset.extraction.traffic.edge_drawers import BaseEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawingParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers import (L2LFeatureParams,
                                                                               LFeatureParams,
                                                                               V2LFeatureParams,
                                                                               V2VFeatureParams,
                                                                               VFeatureParams)
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.feature_computer_container_service import FeatureComputerContainerService
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.defaults import DefaultFeatureComputers
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import T_FeatureComputer
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import T_LikeBaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle

logger = logging.getLogger(__name__)


@dataclass
class TrafficFeatureComputerOptions:
    v: list[T_FeatureComputer[VFeatureParams]] = field(default_factory=DefaultFeatureComputers.v)
    v2v: list[T_FeatureComputer[V2VFeatureParams]] = field(default_factory=DefaultFeatureComputers.v2v)
    l: list[T_FeatureComputer[LFeatureParams]] = field(default_factory=DefaultFeatureComputers.l)
    l2l: list[T_FeatureComputer[L2LFeatureParams]] = field(default_factory=DefaultFeatureComputers.l2l)
    v2l: list[T_FeatureComputer[V2LFeatureParams]] = field(default_factory=DefaultFeatureComputers.v2l)
    l2v: list[T_FeatureComputer[V2LFeatureParams]] = field(default_factory=DefaultFeatureComputers.l2v)


@dataclass
class TrafficExtractorOptions(BaseExtractorOptions):
    """
    Configuration options for traffic data extraction.
    """
    edge_drawer: BaseEdgeDrawer
    feature_computers: TrafficFeatureComputerOptions = field(default_factory=TrafficFeatureComputerOptions)
    assign_multiple_lanelets: bool = True
    ego_map_radius: Optional[float] = None
    ego_map_update_freq: int = 10
    ego_map_strict: bool = False
    ignore_unassigned_vehicles: bool = True
    include_vehicle_vertices: bool = True
    include_lanelet_vertices: bool = True
    linear_lanelet_projection: bool = False
    only_ego_inc_edges: bool = False
    postprocessors: Sequence[T_LikeBaseDataPostprocessor] = field(default_factory=list)
    suppress_feature_computer_exceptions: bool = False


@dataclass
class TrafficExtractionParams(BaseExtractionParams):
    ego_vehicle: Optional[EgoVehicle] = None


class TrafficExtractor(
    BaseExtractor[TrafficExtractorOptions, TrafficExtractionParams, CommonRoadData]
):
    """
    TrafficExtractor returns PyTorch-based heterogeneous graph representations of the current traffic scene, 
    including both vehicle and lanelet nodes as well as the associated graph edges. The extracted graph 
    features are fully customizable via specification of the feature computers provided at initialization.
    """

    def __init__(
        self,
        simulation: BaseSimulation,
        options: TrafficExtractorOptions,
    ) -> None:
        super().__init__(simulation=simulation, options=options)
        self._setup(self.simulation.options)
        self._iterator: Optional[Iterator[tuple[int, Scenario]]] = None
        self._lanelet_id_to_idx: dict[int, int]
        self._lanelet_indices: list[int]
        self.update_lanelet_map(
            lanelets=simulation.lanelet_scenario.lanelet_network.lanelets
        )

    def update_lanelet_map(
        self,
        lanelets: list[Lanelet]
    ) -> None:
        self._lanelet_id_to_idx: dict[int, int] = {
            lanelet.lanelet_id: self.simulation.lanelet_id_to_lanelet_idx[lanelet.lanelet_id]
            for lanelet in lanelets
        }
        # Can't use dict values to index tensors
        # IndexError: only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices
        # (got dict_values)
        self._lanelet_indices: list[int] = list(self._lanelet_id_to_idx.values())

    def extract(
        self,
        time_step: int,
        params: Optional[TrafficExtractionParams] = None
    ) -> CommonRoadData:
        r"""
        Extracts a heterogeneous graph representation of vehicles and lanelets in a CommonRoad scenario intended for
        PyTorch geometric.

        Args:
            time_step (int): Time step for which data instance should be extracted.
            params (T_BaseExtractionParams): Additional parameters for extracting the data instance.

        Returns:
            CommonRoadData: Pytorch-Geometric CommonRoadData instance representing the scene.
        """
        params = params or TrafficExtractionParams()
        ego_vehicle = params.ego_vehicle

        if ego_vehicle is not None and self.options.ego_map_radius is not None:
            update_map = (time_step % self.options.ego_map_update_freq == 0) or \
                         self.simulation.has_changed_lanelet(ego_vehicle.as_dynamic_obstacle)

            if update_map:
                ego_map_lanelets = self.simulation.lanelet_scenario.lanelet_network.lanelets_in_proximity(
                    point=ego_vehicle.state.position,
                    radius=self.options.ego_map_radius
                )
                self.update_lanelet_map(lanelets=ego_map_lanelets)

        v_data, current_obstacles = self._extract_v_data(
            time_step=time_step,
            ego_vehicle=ego_vehicle
        )
        l_data = self._extract_l_data(
            time_step=time_step,
            ego_vehicle=ego_vehicle
        )
        v2l_data, l2v_data = self._extract_v2l_l2v_data(
            time_step=time_step,
            v_data=v_data,
            current_obstacles=current_obstacles,
            ego_vehicle=ego_vehicle
        )
        l2l_data = self._extract_l2l_data(
            time_step=time_step,
            ego_vehicle=ego_vehicle,
            l_data=l_data
        )
        v2v_data = self._extract_v2v_data(
            time_step=time_step,
            v_data=v_data,
            l_data=l_data,
            v2l_data=v2l_data,
            current_obstacles=current_obstacles,
            ego_vehicle=ego_vehicle
        )

        data = CommonRoadData(
            scenario_id=str(self.scenario.scenario_id),
            dt=self.scenario.dt,
            time_step=time_step,
            v_data=v_data,
            v_column_indices=self._v_feature_container_service.feature_column_indices,
            v2v_data=v2v_data,
            v2v_column_indices=self._v2v_feature_container_service.feature_column_indices,
            l_data=l_data,
            l_column_indices=self._l_feature_container_service.feature_column_indices,
            l2l_data=l2l_data,
            l2l_column_indices=self._l2l_feature_container_service.feature_column_indices,
            v2l_data=v2l_data,
            v2l_column_indices=self._v2l_feature_container_service.feature_column_indices,
            l2v_data=l2v_data,
            l2v_column_indices=self._l2v_feature_container_service.feature_column_indices,
        )

        if params.device is not None:
            data = data.to(params.device)

        if not params.disable_postprocessing:
            for postprocessor in self.options.postprocessors:
                result = postprocessor(
                    [data],
                    simulation=self.simulation,
                    ego_vehicle=ego_vehicle
                )
                if len(result) == 1:
                    data = result[0]
                else:
                    logger.warning(f"Skipping ambiguous postprocessing routine {type(postprocessor).__name__} "
                                   f"(returned {len(result)} samples)")

        return data

    def _extract_v_data(
        self,
        time_step: int,
        ego_vehicle: Optional[EgoVehicle]
    ) -> Tuple[Dict[str, Any], List[DynamicObstacle]]:

        ego_state = ego_vehicle.state if ego_vehicle is not None else None
        ego_route = ego_vehicle.ego_route if ego_vehicle is not None else None

        current_obstacles: List[DynamicObstacle]
        if self.options.ignore_unassigned_vehicles:
            current_obstacles = []
            for obstacle in self.simulation.current_obstacles:
                lanelet_assignments = self.simulation.obstacle_id_to_lanelet_id.get(obstacle.obstacle_id, [])
                if ego_vehicle is not None and self.options.ego_map_radius is not None:
                    if self.options.ego_map_strict:
                        obstacle_state = state_at_time(obstacle, time_step, assume_valid=True)
                        distance = np.linalg.norm(ego_state.position - obstacle_state.position)
                        if distance > self.options.ego_map_radius:
                            continue
                    lanelet_assignments = [
                        lanelet_id
                        for lanelet_id in lanelet_assignments
                        if lanelet_id in self._lanelet_id_to_idx
                    ]
                # Ignoring non-assigned non-ego vehicles
                if lanelet_assignments or (ego_vehicle is not None and ego_vehicle.obstacle_id == obstacle.obstacle_id):
                    current_obstacles.append(obstacle)
        else:
            current_obstacles = self.simulation.current_obstacles
        num_vehicles = len(current_obstacles)

        # Ego mask (1 if is ego vehicle, otherwise 0)
        is_ego_mask = torch.zeros((num_vehicles, 1), dtype=torch.bool)

        # Vehicle id tensor
        vehicle_id = torch.empty((num_vehicles, 1), dtype=torch.long)

        # Vehicle type tensor
        vehicle_type = torch.empty((num_vehicles, 1), dtype=torch.long)

        # Vehicle state
        pos = torch.empty((num_vehicles, 2), dtype=torch.float32)
        orientation = torch.empty((num_vehicles, 1), dtype=torch.float32)

        # Vehicle vertices
        vertices = torch.empty((num_vehicles, 10), dtype=torch.float32) if self.options.include_vehicle_vertices else None

        # Iterating over all vehicles to define the vehicle node features
        feature_computer_jobs = []

        for obstacle_idx, obstacle in enumerate(current_obstacles):
            obstacle_id = obstacle.obstacle_id
            state = state_at_time(obstacle, time_step, assume_valid=True)
            raw_pos = torch.from_numpy(state.position).to(torch.float32)
            pos[obstacle_idx, :] = raw_pos
            orientation[obstacle_idx, :] = state.orientation
            if self.options.include_vehicle_vertices:
                vertices[obstacle_idx, :] = torch.from_numpy(obstacle.obstacle_shape.vertices).to(torch.float32).flatten()
            vehicle_id[obstacle_idx, :] = obstacle_id
            vehicle_type[obstacle_idx, :] = map_obstacle_type(obstacle.obstacle_type)

            # Defining binary flag indicating whether a vehicle is the ego vehicle (1) or not (0).
            is_ego_vehicle = ego_vehicle is not None and obstacle_id == ego_vehicle.obstacle_id
            if is_ego_vehicle:
                is_ego_mask[obstacle_idx, :] = True

            feature_computer_jobs.append(VFeatureParams(
                dt=self.scenario.dt,
                time_step=time_step,
                obstacle=obstacle,
                state=state,
                is_ego_vehicle=is_ego_vehicle,
                ego_state=ego_state,
                ego_route=ego_route
            ))

        # Vehicle node feature matrix
        x_vehicle = self._v_feature_container_service.compute_all(feature_computer_jobs, self._simulation)

        data = dict(
            x=x_vehicle,
            is_ego_mask=is_ego_mask,
            pos=pos,
            orientation=orientation,
            num_nodes=num_vehicles,
            id=vehicle_id,
            type=vehicle_type
        )
        if self.options.include_vehicle_vertices:
            data['vertices'] = vertices

        return data, current_obstacles

    def _extract_v2v_data(
        self,
        time_step: int,
        ego_vehicle: Optional[EgoVehicle],
        v_data: Dict[str, Any],
        l_data: Dict[str, Any],
        v2l_data: Dict[str, Any],
        current_obstacles: Optional[List[DynamicObstacle]] = None
    ) -> Dict[str, Any]:

        ego_state = ego_vehicle.state if ego_vehicle is not None else None

        # Vehicle-to-vehicle edges
        edge_index, dist_matrix = self.options.edge_drawer(
            options=BaseEdgeDrawingParams(
                simulation=self.simulation,
                pos=v_data['pos'],
                v_data=v_data,
                l_data=l_data,
                v2l_data=v2l_data
            )
        )

        current_obstacles = current_obstacles if current_obstacles is not None else self.simulation.current_obstacles

        if ego_vehicle is not None and self.options.only_ego_inc_edges and v_data['is_ego_mask'].sum().item() > 0:
            assert ego_vehicle.obstacle_id is not None
            for ego_vehicle_idx in torch.where(v_data['is_ego_mask'])[0]:
                ego_edge_mask = edge_index[1] == ego_vehicle_idx
                edge_index = edge_index[:, ego_edge_mask]

        n_edges = edge_index.shape[1]

        # Vehicle distance matrix
        vehicle_distance = torch.empty((n_edges, 1), dtype=torch.float32)

        feature_computer_jobs = []
        for i in range(n_edges):
            from_node: int = int(edge_index[0][i].item())
            to_node: int = int(edge_index[1][i].item())
            distance: float = dist_matrix[from_node, to_node].item()
            vehicle_distance[i] = distance

            source_obstacle = current_obstacles[from_node]
            source_state = state_at_time(source_obstacle, time_step, assume_valid=True)
            source_is_ego_vehicle = ego_vehicle is not None and source_obstacle.obstacle_id == ego_vehicle.obstacle_id

            target_obstacle = current_obstacles[to_node]
            target_state = state_at_time(target_obstacle, time_step, assume_valid=True)
            target_is_ego_vehicle = ego_vehicle is not None and target_obstacle.obstacle_id == ego_vehicle.obstacle_id

            feature_computer_jobs.append(V2VFeatureParams(
                dt=self.scenario.dt,
                time_step=time_step,
                distance=distance,
                source_obstacle=source_obstacle,
                source_state=source_state,
                source_is_ego_vehicle=source_is_ego_vehicle,
                target_obstacle=target_obstacle,
                target_state=target_state,
                target_is_ego_vehicle=target_is_ego_vehicle,
                ego_state=ego_state
            ))

        # Vehicle edge feature matrix
        edge_attr = self._v2v_feature_container_service.compute_all(feature_computer_jobs, self._simulation)

        data = dict(
            edge_index=edge_index,
            edge_attr=edge_attr,
            distance=vehicle_distance
        )

        return data

    def _extract_l_data(
        self,
        time_step: int,
        ego_vehicle: Optional[EgoVehicle],
    ) -> Dict[str, Any]:

        num_lanelets = len(self._lanelet_id_to_idx)
        ego_state = ego_vehicle.state if ego_vehicle is not None else None
        static_data = self._simulation.lanelet_graph_data
        map_filter_active = ego_vehicle is not None and self.options.ego_map_radius is not None

        # Computing lanelet node features
        feature_computer_jobs = []
        if self._l_feature_container_service.num_computers > 0:
            for lanelet_id in self._lanelet_id_to_idx:
                lanelet = self.simulation.lanelet_network.find_lanelet_by_id(lanelet_id)
                feature_computer_jobs.append(LFeatureParams(
                    dt=self.scenario.dt,
                    time_step=time_step,
                    lanelet=lanelet,
                    ego_state=ego_state
                ))
            # Lanelet feature matrix
            x_lanelet = self._l_feature_container_service.compute_all(feature_computer_jobs, self._simulation)
        else:
            x_lanelet = torch.zeros((num_lanelets, 0), dtype=torch.float32)

        data = dict(
            x=x_lanelet,
            num_nodes=num_lanelets
        )

        # Copying attributes
        for key in keys(static_data):
            if key.startswith('edge_'):
                continue
            if static_data[key].ndim > 0:
                if map_filter_active:
                    data[key] = static_data[key][self._lanelet_indices]
                else:
                    data[key] = static_data[key]

        # TODO: O
        if self.options.include_lanelet_vertices:
            if 'right_vertices' in static_data:
                data.update(dict(
                    right_vertices=static_data.right_vertices[self._lanelet_indices].flatten(start_dim=1),
                    left_vertices=static_data.left_vertices[self._lanelet_indices].flatten(start_dim=1),
                    center_vertices=static_data.center_vertices[self._lanelet_indices].flatten(start_dim=1),
                ))
            if 'relative_vertices' in static_data:
                data.update(dict(
                    relative_vertices=static_data.relative_vertices[self._lanelet_indices].flatten(start_dim=1),
                    vertices_lengths=static_data.vertices_lengths[self._lanelet_indices].flatten(start_dim=1),
                ))

        return data

    def _extract_v2l_l2v_data(
        self,
        time_step: int,
        v_data: Dict[str, Any],
        ego_vehicle: Optional[EgoVehicle],
        current_obstacles: Optional[List[DynamicObstacle]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        ego_state = ego_vehicle.state if ego_vehicle is not None else None

        # Vehicle-to-lanelet edges
        edge_index_v2l = []

        # Computing vehicle-to-lanelet & lanelet-to-vehicle edge features
        feature_computer_jobs = []

        has_arclength = V_Feature.LaneletArclengthRel.value in self._v_feature_container_service.feature_dimensions
        if has_arclength:
            feature_column_indices = self._v_feature_container_service.feature_column_indices
            arclength_rel_column_index = feature_column_indices[V_Feature.LaneletArclengthRel.value][0]
            arclength_abs_column_index = feature_column_indices[V_Feature.LaneletArclengthAbs.value][0]
        else:
            arclength_rel_column_index = None
            arclength_abs_column_index = None

        arclength_rel = []
        arclength_abs = []

        current_obstacles = current_obstacles if current_obstacles is not None else self.simulation.current_obstacles

        for obstacle_idx, obstacle in enumerate(current_obstacles):
            obstacle_id = obstacle.obstacle_id

            is_ego_vehicle = ego_vehicle is not None and obstacle.obstacle_id == ego_vehicle.obstacle_id

            # TODO: Cleanup this section
            lanelet_assignments: List[int]
            if ego_vehicle is not None and self.options.ego_map_radius is not None:
                # TODO: very inefficient
                lanelet_assignments = [
                    lanelet_id
                    for lanelet_id in self.simulation.obstacle_id_to_lanelet_id[obstacle_id]
                    if lanelet_id in self._lanelet_id_to_idx
                ]
            else:
                lanelet_assignments = self.simulation.obstacle_id_to_lanelet_id[obstacle_id]

            for assignment_idx, lanelet_id in enumerate(lanelet_assignments):
                lanelet = self.simulation.find_lanelet_by_id(lanelet_id)
                lanelet_idx = self._lanelet_id_to_idx[lanelet_id]
                state = state_at_time(obstacle, time_step, assume_valid=True)

                if has_arclength and assignment_idx == 0:
                    obstacle_arclength_rel = v_data['x'][obstacle_idx, arclength_rel_column_index].item()
                    obstacle_arclength_abs = v_data['x'][obstacle_idx, arclength_abs_column_index].item()
                else:
                    lanelet_polyline = self.simulation.get_lanelet_center_polyline(lanelet.lanelet_id)
                    obstacle_arclength_abs = lanelet_polyline.get_projected_arclength(
                        state.position,
                        relative=False,
                        linear_projection=self.options.linear_lanelet_projection
                    )
                    obstacle_arclength_rel = obstacle_arclength_abs / lanelet_polyline.length

                arclength_rel.append(obstacle_arclength_rel)
                arclength_abs.append(obstacle_arclength_abs)
                edge_index_v2l.append([obstacle_idx, lanelet_idx])

                feature_computer_jobs.append(V2LFeatureParams(
                    dt=self.scenario.dt,
                    time_step=time_step,
                    obstacle=obstacle,
                    state=state,
                    is_ego_vehicle=is_ego_vehicle,
                    lanelet=lanelet,
                    ego_state=ego_state
                ))

                if not self.options.assign_multiple_lanelets:
                    break

        # Vehicle-to-lanelet edge feature matrix
        edge_attr_v2l_th = self._v2l_feature_container_service.compute_all(
            feature_computer_jobs,
            self._simulation
        )

        # Lanelet-to-vehicle edge feature matrix
        edge_attr_l2v_th = self._l2v_feature_container_service.compute_all(
            feature_computer_jobs,
            self._simulation
        )

        if edge_index_v2l:
            edge_index_v2l_th = torch.tensor(edge_index_v2l, dtype=torch.long).T
        else:
            edge_index_v2l_th = torch.zeros((2, 0), dtype=torch.long)
        edge_index_l2v_th = edge_index_v2l_th.flip(dims=(0,))

        arclength_rel_th = torch.tensor(arclength_rel, dtype=torch.float).unsqueeze(-1)
        arclength_abs_th = torch.tensor(arclength_abs, dtype=torch.float).unsqueeze(-1)

        data_v2l = dict(
            edge_index=edge_index_v2l_th,
            edge_attr=edge_attr_v2l_th,
            arclength_rel=arclength_rel_th,
            arclength_abs=arclength_abs_th
        )
        data_l2v = dict(
            edge_index=edge_index_l2v_th,
            edge_attr=edge_attr_l2v_th,
            arclength_rel=arclength_rel_th,
            arclength_abs=arclength_abs_th
        )
        return data_v2l, data_l2v

    def _extract_l2l_data(
        self,
        time_step: int,
        ego_vehicle: Optional[EgoVehicle],
        l_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        ego_state = ego_vehicle.state if ego_vehicle is not None else None

        # Lanelet-to-lanelet edges
        static_data = self.simulation.lanelet_graph_data
        edge_index = static_data.edge_index
        n_edges_total = edge_index.shape[1]

        # Computing lanelet-to-lanelet edge features
        feature_computer_jobs = []

        map_filter_active = ego_vehicle is not None and self.options.ego_map_radius is not None
        included_edge_indices = []

        for i in range(n_edges_total):
            from_node: int = edge_index[0][i].item()
            to_node: int = edge_index[1][i].item()
            from_lanelet_id = self.simulation.lanelet_graph_data.id[from_node].item()
            to_lanelet_id = self.simulation.lanelet_graph_data.id[to_node].item()

            if map_filter_active:
                if from_lanelet_id in self._lanelet_id_to_idx and to_lanelet_id in self._lanelet_id_to_idx:
                    included_edge_indices.append(i)

            source_lanelet = self.simulation.find_lanelet_by_id(from_lanelet_id)
            target_lanelet = self.simulation.find_lanelet_by_id(to_lanelet_id)

            feature_computer_jobs.append(L2LFeatureParams(
                dt=self.scenario.dt,
                time_step=time_step,
                source_lanelet=source_lanelet,
                target_lanelet=target_lanelet,
                ego_state=ego_state
            ))

        # Lanelet-to-lanelet edge feature matrix
        edge_attr = self._l2l_feature_container_service.compute_all(feature_computer_jobs, self._simulation)

        if map_filter_active:
            edge_index, _, edge_mask = subgraph(
                subset=self._lanelet_indices,
                edge_index=edge_index,
                edge_attr=None,
                relabel_nodes=True,
                return_edge_mask=True
            )

        data = dict(
            edge_attr=edge_attr[edge_mask] if map_filter_active else edge_attr,
            edge_index=edge_index,
            num_edges=edge_index.shape[1]
        )

        # Copying static edge attributes
        for key in keys(static_data):
            if not key.startswith('edge_attr_'):
                continue
            new_key = lchop(key, "edge_attr_")
            if static_data[key].ndim > 0:
                if map_filter_active:
                    data[new_key] = static_data[key][edge_mask]
                else:
                    data[new_key] = static_data[key]

        return data

    def reset_feature_computers(self) -> None:
        self._v_feature_container_service.reset_all_feature_computers(self._simulation)
        self._v2v_feature_container_service.reset_all_feature_computers(self._simulation)
        self._l_feature_container_service.reset_all_feature_computers(self._simulation)
        self._l2l_feature_container_service.reset_all_feature_computers(self._simulation)
        self._v2l_feature_container_service.reset_all_feature_computers(self._simulation)
        BaseFeatureComputer.reset_cache()

    def _setup(self, simulation_options: BaseSimulationOptions) -> None:
        # vehicle node features
        self._v_feature_container_service: FeatureComputerContainerService[VFeatureParams] = \
            FeatureComputerContainerService(
                feature_computers=self.options.feature_computers.v,
                suppress_feature_computer_exceptions=self.options.suppress_feature_computer_exceptions,
                simulation_options=simulation_options
            )

        # vehicle to vehicle edge features
        self._v2v_feature_container_service: FeatureComputerContainerService[V2VFeatureParams] = \
            FeatureComputerContainerService(
                self.options.feature_computers.v2v,
                suppress_feature_computer_exceptions=self.options.suppress_feature_computer_exceptions,
                simulation_options=simulation_options
            )

        # lanelet node features
        self._l_feature_container_service: FeatureComputerContainerService[LFeatureParams] = \
            FeatureComputerContainerService(
                self.options.feature_computers.l,
                suppress_feature_computer_exceptions=self.options.suppress_feature_computer_exceptions,
                simulation_options=simulation_options
            )

        # lanelet to lanelet edge features
        self._l2l_feature_container_service: FeatureComputerContainerService[L2LFeatureParams] = \
            FeatureComputerContainerService(
                self.options.feature_computers.l2l,
                suppress_feature_computer_exceptions=self.options.suppress_feature_computer_exceptions,
                simulation_options=simulation_options
            )

        # vehicle to lanelet edge features
        self._v2l_feature_container_service: FeatureComputerContainerService[V2LFeatureParams] = \
            FeatureComputerContainerService(
                self.options.feature_computers.v2l,
                suppress_feature_computer_exceptions=self.options.suppress_feature_computer_exceptions,
                simulation_options=simulation_options
            )

        # lanelet to vehicle edge features
        self._l2v_feature_container_service: FeatureComputerContainerService[V2LFeatureParams] = \
            FeatureComputerContainerService(
                self.options.feature_computers.l2v,
                suppress_feature_computer_exceptions=self.options.suppress_feature_computer_exceptions,
                simulation_options=simulation_options
            )

        # Important: Reset of feature computers needs to switch back to self._simulation from dummy_simulation
        self.reset_feature_computers()

    def __iter__(self) -> BaseExtractor:
        self._iterator = iter(self._simulation)
        return self

    def __next__(self) -> CommonRoadData:
        if self._iterator is None:
            raise StopIteration()
        try:
            time_step, scenario = next(self._iterator)
            data = self.extract(time_step=time_step, params=TrafficExtractionParams())
            return data
        except StopIteration as e:
            self._iterator = None
            raise e

    def __len__(self) -> T_CountParam:
        return self.simulation.num_time_steps
