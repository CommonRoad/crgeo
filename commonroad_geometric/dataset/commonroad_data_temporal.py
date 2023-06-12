from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple, Type, Union, cast, overload

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.data.collate import collate
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.separate import _separate, separate
from torch_geometric.typing import EdgeType, NodeType

from commonroad_geometric.common.torch_utils.helpers import get_index_mapping_by_first_occurrence
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.commonroad_data import CommonRoadData, VirtualAttributesEdgeStorage, VirtualAttributesNodeStorage

if TYPE_CHECKING:
    from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VTVFeatureParams


# We don't directly inherit from torch_geometric.data.batch.Batch to avoid the DynamicInheritance metaclass
class CommonRoadDataTemporal(CommonRoadData):
    """A CommonRoadData heterogeneous temporal graph.
    Spatio-temporal graphs encode spatial relationships as well as a time dimension in the graph structure.
    For example, nodes can be associated with a time step and edges can encode relative positions of connected nodes.

    Please note that creating a batch of CommonRoadDataTemporal instances comes with some problems.
    For more information check out https://gitlab.lrz.de/cps/commonroad-geometric/-/issues/229.
    """

    @classmethod
    def from_data_list(
        cls,
        data_list: List[CommonRoadData],
        delta_time: float,
    ) -> CommonRoadDataTemporal:
        # based on torch_geometric.data.batch.Batch.from_data_list
        temporal_graph: CommonRoadDataTemporal
        temporal_graph, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=True,
            follow_batch=None,
            exclude_keys=None,
        )
        temporal_graph._num_graphs = len(data_list)
        temporal_graph._slice_dict = slice_dict
        temporal_graph._inc_dict = inc_dict

        temporal_graph.delta_time = delta_time
        return temporal_graph

    @staticmethod
    def add_temporal_vehicle_edges_(
        data: CommonRoadDataTemporal,
        max_time_steps_temporal_edge: T_CountParam,
        *,
        obstacle_id_to_obstacle_idx: Optional[List[Dict[int, int]]] = None,
        feature_computers: Optional[Sequence[Callable[[VTVFeatureParams], List[float]]]] = None,
    ) -> None:
        from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VTVFeatureParams
        T = data.num_graphs
        device = data.vehicle.x.device

        data.max_time_steps_temporal_edge = None if max_time_steps_temporal_edge is Unlimited \
            else max_time_steps_temporal_edge

        vehicle_batch_idx_incs = data._slice_dict["vehicle"]["x"][:-1].tolist()

        if obstacle_id_to_obstacle_idx is None:
            obstacle_id_to_obstacle_idx = [
                {
                    obstacle_id.item(): obstacle_idx
                    for obstacle_idx, obstacle_id in enumerate(data.storage_attribute_at_time_step("vehicle", "id", time_step).view(-1))
                }
                for time_step in range(T)
            ]
        assert len(obstacle_id_to_obstacle_idx) == T

        # all obstacle ids across all time steps
        obstacle_ids = set(
            obstacle_id
            for obstacle_mapping in obstacle_id_to_obstacle_idx
            for obstacle_id in obstacle_mapping.keys()
        )

        obstacle_id_to_t_ind: Dict[int, List[Optional[int]]] = {}
        # obstacle_id_to_t_ind[1][2] is the index of obstacle 1 at time step 2 in the combined graph
        # it is None if the obstacle did not exist at time step 2
        for obstacle_id in obstacle_ids:
            obstacle_id_to_t_ind[obstacle_id] = []
            for time_step, inc in zip(range(T), vehicle_batch_idx_incs):
                idx = obstacle_id_to_obstacle_idx[time_step].get(obstacle_id, None)
                obstacle_id_to_t_ind[obstacle_id].append(
                    idx + inc if idx is not None else None
                )

        # ("vehicle", "temporal", "vehicle") edges
        edge_index: List[Tuple[int, int]] = []
        edge_attr: List[List[float]] = []
        edge_t_src: List[List[int]] = []
        edge_attr_indices_mapping = {
            "delta_time": (0, 1)
        }
        num_edge_attr = 1

        # create directed temporal edges for each vehicle
        # temporal edges connect a vehicle node to all vehicle nodes of itself at later time steps,
        # up to max_time_steps_temporal_edge time steps away
        for time_step in range(1, T):
            for obstacle_id in obstacle_id_to_obstacle_idx[time_step].keys():
                obstacle_t_ind = obstacle_id_to_t_ind[obstacle_id]
                curr_obstacle_idx = obstacle_t_ind[time_step]

                if max_time_steps_temporal_edge is Unlimited:
                    max_ts_offset = time_step
                else:
                    max_ts_offset = min(time_step, max_time_steps_temporal_edge)

                for ts_offset in range(1, max_ts_offset + 1):
                    past_obstacle_idx = obstacle_t_ind[time_step - ts_offset]
                    if past_obstacle_idx is not None:
                        edge_index.append((past_obstacle_idx, curr_obstacle_idx))

                        # compute edge attributes
                        delta_time_edge = ts_offset * data.delta_time
                        curr_edge_attr_list = [delta_time_edge]

                        params = VTVFeatureParams(
                            dt=data.delta_time,
                            data=data,
                            time_step=time_step,
                            past_obstacle_idx=past_obstacle_idx,
                            curr_obstacle_idx=curr_obstacle_idx,
                        )

                        if feature_computers is not None:
                            for feature_computer in feature_computers:
                                feature_dict = feature_computer(params)
                                for k, v in feature_dict.items():
                                    if k not in edge_attr_indices_mapping:
                                        feature_size = v.shape[0] if isinstance(v, Tensor) else 1 # TODO
                                        edge_attr_indices_mapping[k] = (num_edge_attr, num_edge_attr + feature_size)
                                        num_edge_attr += feature_size

                                    curr_edge_attr_list.extend(v.tolist() if isinstance(v, Tensor) else [v])
                        edge_attr.append(curr_edge_attr_list)

                        edge_t_src.append([time_step])

        data._edge_store_dict["vehicle", "temporal", "vehicle"] = VirtualAttributesEdgeStorage(
            parent=data,
            key=("vehicle", "temporal", "vehicle"),
            feature_key="edge_attr",
            column_indices_mapping=edge_attr_indices_mapping,
        )
        data["vehicle", "temporal", "vehicle"].edge_index = torch.tensor(
            edge_index,
            dtype=torch.long,
            device=device,
        ).t().contiguous()
        data["vehicle", "temporal", "vehicle"].edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        data["vehicle", "temporal", "vehicle"].t_src = torch.tensor(edge_t_src, dtype=torch.long, device=device)

    @staticmethod
    def remove_duplicate_lanelet_graphs_(data: CommonRoadDataTemporal) -> None:
        N = data.num_graphs
        if N == 1:
            return  # no duplicate graphs

        num_unique_nodes = data.lanelet.num_nodes // N
        num_unique_edges = data.lanelet_to_lanelet.num_edges // N

        # check assumptions
        assert (torch.diff(data._slice_dict["lanelet"]["x"]) == num_unique_nodes).all()  # equal number of nodes in each graph
        # same node attributes
        for i in range(1, N):
            assert (data.lanelet.x[:num_unique_nodes] == data.lanelet.x[i*num_unique_nodes:(i+1)*num_unique_nodes]).all()
        assert (torch.diff(data._inc_dict["lanelet", "to", "lanelet"]["edge_index"][:, 0, 0]) == num_unique_nodes).all()  # equal number of nodes in each graph
        assert (torch.diff(data._slice_dict["lanelet", "to", "lanelet"]["edge_index"]) == num_unique_edges).all()  # equal number of edges in each graph
        # same edge order & edge attributes
        for i in range(1, N):
            assert (data.lanelet_to_lanelet.edge_index[:, :num_unique_edges] == (data.lanelet_to_lanelet.edge_index[:, i*num_unique_edges:(i+1)*num_unique_edges] % num_unique_nodes)).all()
            assert (data.lanelet_to_lanelet.edge_attr[:num_unique_edges] == (data.lanelet_to_lanelet.edge_attr[i*num_unique_edges:(i+1)*num_unique_edges])).all()

        # replace duplicate lanelet edge indices
        data.lanelet_to_vehicle.edge_index[0] = data.lanelet_to_vehicle.edge_index[0] % num_unique_nodes
        data.vehicle_to_lanelet.edge_index[1] = data.vehicle_to_lanelet.edge_index[1] % num_unique_nodes
        data._inc_dict["lanelet", "to", "vehicle"]["edge_index"][:, 0] = 0
        data._inc_dict["vehicle", "to", "lanelet"]["edge_index"][:, 1] = 0

        # remove duplicate lanelet nodes and edges
        # see torch_geometric/data/separate.py
        for key in ["lanelet", ("lanelet", "to", "lanelet")]:
            for attr in data._slice_dict[key].keys():
                data[key][attr] = _separate(
                    key=attr,
                    value=data[key][attr],
                    idx=0,
                    slices=data._slice_dict[key][attr],
                    incs=data._inc_dict[key][attr],
                    batch=data,
                    store=data[key],
                    decrement=True,
                )
        data.lanelet.num_nodes = data.lanelet._num_nodes[0]

        data._duplicate_lanelet_graphs_attrs_dict = {
            "lanelet": list(data._slice_dict["lanelet"].keys()),
            ("lanelet", "to", "lanelet"): list(data._slice_dict["lanelet", "to", "lanelet"].keys()),
        }
        del data._slice_dict["lanelet"]
        del data._inc_dict["lanelet"]
        del data._slice_dict["lanelet", "to", "lanelet"]
        del data._inc_dict["lanelet", "to", "lanelet"]

    def __init__(self, _base_cls: Optional[Type] = None) -> None:
        assert _base_cls is None or _base_cls is CommonRoadData
        super().__init__()

    # "manually inherit" methods from torch_geometric.data.batch.Batch
    index_select = Batch.index_select
    to_data_list = Batch.to_data_list
    num_graphs = Batch.num_graphs
    # exclude __reduce__

    def get_example(self, idx: int) -> CommonRoadData:
        data = cast(CommonRoadData, separate(
            cls=CommonRoadData,
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        ))
        if hasattr(self, "_duplicate_lanelet_graphs_attrs_dict"):
            for key, attrs in self._duplicate_lanelet_graphs_attrs_dict.items():
                for attr in attrs:
                    data[key][attr] = self[key][attr]

        return data

    def __getitem__(self, idx: Union[NodeType, EdgeType, int, np.integer, IndexType]) -> Any:
        # we must manually handle the case which gets delegated to super().__getitem__
        if isinstance(idx, str) or isinstance(idx, tuple) and len(idx) > 0 and isinstance(idx[0], str):
            return super().__getitem__(idx)
        return Batch.__getitem__(self, idx)

    def get_time_window(self, time_slice: slice) -> CommonRoadDataTemporal:
        start, end, stride = time_slice.indices(self.num_graphs)
        assert stride == 1 and start < end
        if start == 0 and end == self.num_graphs:
            return self

        device = self.vehicle.x.device
        data_list = cast(List[CommonRoadData], self.index_select(slice(start, end)))
        data_slice = CommonRoadDataTemporal.from_data_list(
            data_list=data_list,
            delta_time=self.delta_time,
        )

        if end - start > 1 and self.vehicle_temporal_vehicle is not None:
            # reconstruct edge_index, edge_attr, t_src attributes of ("vehicle", "temporal", "vehicle") edges
            time_steps = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
            time_slice_mask = torch.any(self.vehicle_temporal_vehicle.t_src == time_steps, dim=1)
            data_slice.vehicle_temporal_vehicle.edge_attr = self.vehicle_temporal_vehicle.edge_attr[time_slice_mask]
            data_slice.vehicle_temporal_vehicle.t_src = self.vehicle_temporal_vehicle.t_src[time_slice_mask]

            if start > 0:
                edge_index_start = self.vehicle.ptr[start]
                edge_index_inv_mask = torch.any(self.vehicle_temporal_vehicle.edge_index < edge_index_start, dim=0)
            if end < self.num_graphs:
                edge_index_end = self.vehicle.ptr[end]
                edge_index_mask_end = torch.any(self.vehicle_temporal_vehicle.edge_index >= edge_index_end, dim=0)
                if start > 0:
                    edge_index_inv_mask = torch.logical_or(edge_index_inv_mask, edge_index_mask_end)
                else:
                    edge_index_inv_mask = edge_index_mask_end

            data_slice.vehicle_temporal_vehicle.edge_index = self.vehicle_temporal_vehicle.edge_index[:, ~edge_index_inv_mask]
            if start > 0:
                data_slice.vehicle_temporal_vehicle.edge_index -= edge_index_start

        return data_slice

    @property
    def vehicle_temporal_vehicle(self) -> Optional[VirtualAttributesEdgeStorage]:
        return self._edge_store_dict.get(("vehicle", "temporal", "vehicle"))

    @property
    def vtv(self) -> Optional[VirtualAttributesEdgeStorage]:
        return self._edge_store_dict.get(("vehicle", "temporal", "vehicle"))

    @overload
    def storage_at_time_step(self, key: str, time_step: int) -> VirtualAttributesNodeStorage:
        ...

    @overload
    def storage_at_time_step(self, key: Tuple[str, str, str], time_step: int) -> VirtualAttributesEdgeStorage:
        ...

    def storage_at_time_step(
        self,
        key: Union[str, Tuple[str, str, str]],
        time_step: int,
    ) -> Union[VirtualAttributesNodeStorage, VirtualAttributesEdgeStorage]:
        is_node_store = isinstance(key, str)
        cls = VirtualAttributesNodeStorage if is_node_store else VirtualAttributesEdgeStorage
        storage = self[key]
        storage_t = cls(
            parent=self,
            key=key,
            feature_key=storage._feature_key,
            column_indices_mapping=storage._column_indices,
            mask_key=storage._mask_key,
        )

        attrs = self._slice_dict[key].keys()
        for attr in attrs:
            storage_t[attr] = self.storage_attribute_at_time_step(key=key, attr=attr, time_step=time_step)

        return storage_t

    def storage_attribute_at_time_step(
        self,
        key: Union[str, Tuple[str, str, str]],
        attr: str,
        time_step: int,
    ) -> Any:
        return _separate(
            key=key,
            value=self[key][attr],
            idx=time_step,
            slices=self._slice_dict[key][attr],
            incs=self._inc_dict[key][attr],
            batch=self,
            store=self[key],
            decrement=True,
        )

    def vehicle_at_time_step(self, t: int) -> VirtualAttributesNodeStorage:
        return self.storage_at_time_step("vehicle", t)

    def get_node_features_temporal_sequence_for_vehicle(self, vehicle_id: int, keys: Optional[Sequence[str]] = None) -> Tensor:
        # TODO: Document, rename & unit-test
        if self.vehicle is None:
            raise AttributeError("self.vehicle is None")
        if keys is None:
            x = self.vehicle.x # type: ignore
        else:
            x = torch.cat([self.vehicle[key] for key in keys], dim=-1)
        id = self.vehicle.id.squeeze() # type: ignore
        x_v: Tensor = x[id==vehicle_id, :]
        return x_v

    def get_node_features_temporal_sequence(self, keys: Optional[Sequence[str]] = None) -> Tensor:
        # TODO: Document, rename & unit-test
        if self.vehicle is None:
            raise AttributeError("self.vehicle is None")
        if keys is None:
            x = self.vehicle.x
        else:
            x = torch.cat([self.vehicle[key] for key in keys], dim=-1)
        n_features = x.shape[1]
        n_timesteps = self.vehicle.batch.max().item() + 1
        indices = get_index_mapping_by_first_occurrence(self.vehicle.id.squeeze(-1))[None, :, None]
        n_vehicles = indices.max().item() + 1
        batch = self.vehicle.batch.unsqueeze(-1)
        x_v = torch.zeros((n_vehicles, x.shape[0], n_features))
        x_v.scatter_(0, indices.repeat(1, 1, n_features), x.unsqueeze(0))
        x_t = torch.zeros((n_vehicles, n_timesteps, n_features))
        x_t.scatter_add_(1, batch.unsqueeze(0).repeat(n_vehicles, 1, n_features), x_v)
        return x_t

    def __reduce__(self) -> Tuple:
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        return CommonRoadDataTemporal, (), self.__getstate__()

class CommonRoadDataTemporalBatch(Batch):
    """
    modified from torch_geometric.data.Batch
    additionally maintaining slice_dict, inc_dict, num_graphs for CommonroadDataTemporal
    """
    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):

        temporal_data_num_graphs = [data._num_graphs for data in data_list]
        temporal_data_slice_dict = [data._slice_dict for data in data_list]
        temporal_data_inc_dict = [data._inc_dict for data in data_list]
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=True,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict
        batch._temporal_data_num_graphs = temporal_data_num_graphs
        batch._temporal_data_slice_dict = temporal_data_slice_dict
        batch._temporal_data_inc_dict = temporal_data_inc_dict

        return batch
    def get_example(self, idx: int) -> BaseData:

        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))
        
        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )
        data._slice_dict = self._temporal_data_slice_dict[idx]
        data._inc_dict = self._temporal_data_inc_dict[idx]
        data._num_graphs = self._temporal_data_num_graphs[idx]
        data.batch_size = self._temporal_data_num_graphs[idx]

        return data