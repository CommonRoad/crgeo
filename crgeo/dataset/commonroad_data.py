from __future__ import annotations

from itertools import chain
import logging
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING


import torch
import torch_geometric.data
import torch_geometric.data.storage
import torch_geometric.data.view
HeteroData = torch_geometric.data.HeteroData
BaseStorage = torch_geometric.data.storage.BaseStorage
MappingView = torch_geometric.data.view.MappingView
ItemsView = torch_geometric.data.view.ItemsView
ValuesView = torch_geometric.data.view.ValuesView
KeysView = torch_geometric.data.view.KeysView
NodeStorage = torch_geometric.data.storage.NodeStorage
EdgeStorage = torch_geometric.data.storage.EdgeStorage

T_ColumnIndicesMapping = Dict[str, Tuple[int, int]]


logger = logging.getLogger(__name__)


class CommonRoadData(HeteroData):

    def __init__(
        self,
        scenario_id: Optional[str] = None,
        dt: Optional[float] = None,
        time_step: Optional[int] = None,
        v_data: Optional[Dict[str, Any]] = None,
        v_column_indices: Optional[T_ColumnIndicesMapping] = None,
        v2v_data: Optional[Dict[str, Any]] = None,
        v2v_column_indices: Optional[T_ColumnIndicesMapping] = None,
        v2v_temporal_data: Optional[Dict[str, Any]] = None,
        v2v_temporal_column_indices: Optional[T_ColumnIndicesMapping] = None,
        l_data: Optional[Dict[str, Any]] = None,
        l_column_indices: Optional[T_ColumnIndicesMapping] = None,
        l2l_data: Optional[Dict[str, Any]] = None,
        l2l_column_indices: Optional[T_ColumnIndicesMapping] = None,
        v2l_data: Optional[Dict[str, Any]] = None,
        v2l_column_indices: Optional[T_ColumnIndicesMapping] = None,
        l2v_data: Optional[Dict[str, Any]] = None,
        l2v_column_indices: Optional[T_ColumnIndicesMapping] = None,
    ):
        super().__init__()

        store_data = [
            (v_data, v_column_indices, "vehicle", "x"),
            (v2v_data, v2v_column_indices, ("vehicle", "to", "vehicle"), "edge_attr"),
            (v2v_temporal_data, v2v_temporal_column_indices, ("vehicle", "temporal", "vehicle"), "edge_attr"),
            (l_data, l_column_indices, "lanelet", "x"),
            (l2l_data, l2l_column_indices, ("lanelet", "to", "lanelet"), "edge_attr"),
            (v2l_data, v2l_column_indices, ("vehicle", "to", "lanelet"), "edge_attr"),
            (l2v_data, l2v_column_indices, ("lanelet", "to", "vehicle"), "edge_attr"),
        ]
        for data, column_indices, key, feature_key in store_data:
            if data is not None:
                assert column_indices is not None
                is_node_data = isinstance(key, str)
                cls = VirtualAttributesNodeStorage if is_node_data else VirtualAttributesEdgeStorage
                attr = self._node_store_dict if is_node_data else self._edge_store_dict
                attr[key] = cls(
                    parent=self,
                    key=key,
                    feature_key=feature_key,
                    column_indices_mapping=column_indices,
                )
                self[key].update(data)

        self._global_store.scenario_id = scenario_id
        self._global_store.dt = dt
        self._global_store.time_step = time_step

    def to_dict(self, include_virtual_attrs: bool = True) -> Dict[str, Any]:
        out = self._global_store.to_dict()
        for store_key, store in chain(self._node_store_dict.items(), self._edge_store_dict.items()):
            for key, value in store.to_dict(include_virtual_attrs=include_virtual_attrs).items():
                if isinstance(store_key, tuple):
                    store_key_str = store_key[0] + '-to-' + store_key[2]
                else:
                    store_key_str = store_key
                out[f"{store_key_str}-{key}"] = value
        return out

    @property
    def store_sizes(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for store in self.node_stores:
            out[store.name] = store.num_nodes
        for store in self.edge_stores:
            out[store.name] = store.num_edges
        return out

    @property
    def device(self) -> torch.device:
        return self.l2l.edge_index.device

    @property
    def scenario_id(self) -> Optional[str]:
        # _global_storage does not store None values
        return self._global_store.get("scenario_id", None)

    @property
    def dt(self) -> Optional[float]:
        return self._global_store.get("dt", None)

    @property
    def time_step(self) -> Optional[int]:
        # _global_storage does not store None values
        return self._global_store.get("time_step", None)

    @property
    def batch_size(self) -> int:
        return int(self.v.batch.max().item()) + 1

    @property
    def vehicle(self) -> VirtualAttributesNodeStorage:
        return self._node_store_dict.get("vehicle")

    @property
    def v(self) -> VirtualAttributesNodeStorage:
        return self._node_store_dict.get("vehicle")

    @property
    def ego(self) -> VirtualAttributesNodeStorage:
        if "_ego_vehicle" not in self.__dict__:
            if self.vehicle is not None and "is_ego_mask" in self.vehicle:
                # ego_vehicle NodeStorage instance shares the underlying storage with vehicle NodeStorage.
                # The only difference to vehicle is mask_key="is_ego_mask".
                # It is not stored in self._node_store_dict so that it is ignored when merging multiple CommonRoadData
                # instances into a batch.
                self.__dict__["_ego_vehicle"] = VirtualAttributesNodeStorage(
                    parent=self,
                    key="vehicle",
                    feature_key="x",
                    column_indices_mapping=self.vehicle._column_indices,
                    mask_key="is_ego_mask",
                )
                self.__dict__["_ego_vehicle"]._mapping = self.vehicle._mapping
            else:
                self.__dict__["_ego_vehicle"] = None

        return self._ego_vehicle

    @property
    def vehicle_to_vehicle(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("vehicle", "to", "vehicle"))

    @property
    def v2v(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("vehicle", "to", "vehicle"))

    @property
    def lanelet(self) -> VirtualAttributesNodeStorage:
        return self._node_store_dict.get("lanelet")

    @property
    def l(self) -> VirtualAttributesNodeStorage:
        return self._node_store_dict.get("lanelet")

    @property
    def lanelet_to_lanelet(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("lanelet", "to", "lanelet"))

    @property
    def l2l(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("lanelet", "to", "lanelet"))

    @property
    def vehicle_to_lanelet(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("vehicle", "to", "lanelet"))

    @property
    def v2l(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("vehicle", "to", "lanelet"))

    @property
    def lanelet_to_vehicle(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("lanelet", "to", "vehicle"))

    @property
    def l2v(self) -> VirtualAttributesEdgeStorage:
        return self._edge_store_dict.get(("lanelet", "to", "vehicle"))

    def flatten(
        self,
        padding: int
    ) -> Dict[str, torch.Tensor]:
        from crgeo.common.torch_utils.helpers import flatten_data
        flattened_data = flatten_data(self, padding=padding)
        return flattened_data

    @classmethod
    def reconstruct(
        cls,
        tensors: Dict[str, torch.Tensor]
    ) -> CommonRoadData:
        from crgeo.common.torch_utils.helpers import reconstruct_data
        hetero_data = reconstruct_data(
            tensors=tensors
        )
        data = CommonRoadData(
            v_data=hetero_data["vehicle"],
            v_column_indices={},
            v2v_data=hetero_data["vehicle", "to", "vehicle"],
            v2v_column_indices={},
            l_data=hetero_data["lanelet"],
            l_column_indices={},
            l2l_data=hetero_data["lanelet", "to", "lanelet"],
            l2l_column_indices={},
            v2l_data=hetero_data["vehicle", "to", "lanelet"],
            v2l_column_indices={},
            l2v_data=hetero_data["lanelet", "to", "vehicle"],
            l2v_column_indices={},
        )
        for k, v in hetero_data._global_store.items():
            data[k] = v
        return data

    def stores_as(self, data: CommonRoadData) -> CommonRoadData:
        # used (among other things) by torch_geometric.data.collate.collate when merging CommonRoadData instances
        # into a batch
        for node_type, node_storage in data._node_store_dict.items():
            if node_type not in self._node_store_dict:
                assert isinstance(node_storage, VirtualAttributesNodeStorage)
                self._node_store_dict[node_type] = VirtualAttributesNodeStorage(
                    parent=self,
                    key=node_type,
                    feature_key=node_storage._feature_key,
                    column_indices_mapping=node_storage._column_indices,
                    mask_key=node_storage._mask_key,
                )

        for edge_type, edge_storage in data._edge_store_dict.items():
            if edge_type not in self._edge_store_dict:
                assert isinstance(edge_storage, VirtualAttributesEdgeStorage)
                self._edge_store_dict[edge_type] = VirtualAttributesEdgeStorage(
                    parent=self,
                    key=edge_type,
                    feature_key=edge_storage._feature_key,
                    column_indices_mapping=edge_storage._column_indices,
                    mask_key=edge_storage._mask_key,
                )

        return self

    def __repr__(self) -> str:
        # TODO show global attrs
        s = f"{type(self).__name__}(scenario_id={self.scenario_id}, t={self.time_step}, dt={self.dt})\n\n"
        s += ' - ' + repr(self.vehicle) + '\n'
        s += ' - ' + repr(self.vehicle_to_vehicle) + '\n'
        try:
            s += ' - ' + repr(self.vehicle_temporal_vehicle) + '\n'
        except AttributeError:
            pass
        s += ' - ' + repr(self.lanelet) + '\n'
        s += ' - ' + repr(self.lanelet_to_lanelet) + '\n'
        s += ' - ' + repr(self.vehicle_to_lanelet) + '\n'
        s += ' - ' + repr(self.lanelet_to_vehicle) + '\n'
        return s


class VirtualAttributesBaseStorage(BaseStorage):

    def __init__(
        self,
        parent: CommonRoadData,
        key: Union[str, Tuple[str, str, str]],
        feature_key: str,
        column_indices_mapping: T_ColumnIndicesMapping,
        mask_key: Optional[str] = None,
    ):
        super().__init__(_parent=parent, _key=key)
        self._feature_key = feature_key
        self._column_indices = column_indices_mapping
        self._mask_key = mask_key

    @property
    def key(self) -> str:
        return self._key

    def __getitem__(self, key: str) -> Any:
        try:
            if key in self._column_indices:
                start_index, end_index = self._column_indices[key]
                value = self._mapping[self._feature_key][:, start_index:end_index]
            else:
                value = self._mapping[key]
        except KeyError as e:
            raise KeyError(f"The feature '{key}' is not included in the CommonRoadData instance. Make sure that the corresponding feature computer is enabled.") from e
        if self._mask_key is not None and isinstance(value, torch.Tensor):
            mask = self._mapping[self._mask_key].bool()
            value = value[mask.squeeze(-1)]
        return value

    def keys(self, *args: List[str], virtual_attributes: bool = False) -> Union[torch_geometric.data.view.KeysView, CustomKeysView]:
        if not virtual_attributes:
            return super().keys(*args)
        return CustomKeysView(self, *args)

    def values(self, *args: List[str], virtual_attributes: bool = False) -> Union[torch_geometric.data.view.ValuesView, CustomValuesView]:
        if not virtual_attributes:
            return super().values(*args)
        return CustomValuesView(self, *args)

    def items(self, *args: List[str], virtual_attributes: bool = False) -> Union[torch_geometric.data.view.ItemsView, CustomItemsView]:
        if not virtual_attributes:
            return super().items(*args)
        return CustomItemsView(self, *args)

    @property
    def name(self) -> str:
        if isinstance(self._key, tuple):
            return '-'.join(self._key)
        return self._key

    @property
    def column_indices(self) -> T_ColumnIndicesMapping:
        return self._column_indices

    @property
    def feature_columns(self) -> Collection[str]:
        return self._column_indices.keys()

    @property
    def feature_dimensions(self) -> Dict[str, int]:
        return {k: v[1] - v[0] for k, v in self._column_indices.items()}

    def to_dict(self, include_virtual_attrs: bool = True) -> Dict[str, Any]:
        d = self._mapping.copy()
        if include_virtual_attrs:
            d.update({
                key: self[key]
                for key in self._column_indices.keys()
            })
        return d

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}({self._key})\n'
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                type_string = f"torch.Tensor ({v.shape}, {v.dtype})"
                s += f"{k:>30}: {type_string:<30}\n"
            else:
                s += f"{k:>30}: {str(v):<30}\n"
            if k == self._feature_key:
                for sub_k, sub_column_indices in self._column_indices.items():
                    if v.shape[0] == 0:
                        s += f"{sub_k:>50}: idx {str(sub_column_indices)} \n"
                    else:
                        s += f"{sub_k:>50}: idx {str(sub_column_indices)}, range {self[sub_k].min():.2f} - {self[sub_k].max():.2f}, mean {self[sub_k].mean():.2f}, std {self[sub_k].std():.2f} \n"

        return s


class CustomMappingView(MappingView):

    def __init__(
        self,
        storage: VirtualAttributesBaseStorage,
        *args: str,
    ):
        super().__init__(mapping=storage._mapping, *args)
        self._storage = storage
        self._virtual_keys = set(storage._column_indices.keys())

    def _keys(self) -> Iterable[str]:
        if self._storage._feature_key not in self._mapping:
            # The feature which all virtual attributes return a slice from does not exist thus the virtual attributes
            # also do not exist.
            return super()._keys()

        assert not set(self._mapping.keys()).intersection(self._virtual_keys), \
            "_virtual_keys overlap with _mapping.keys()"
        if len(self._args) == 0:
            return list(self._mapping.keys()) + list(self._virtual_keys)
        else:
            keys = self._virtual_keys.union(self._mapping.keys())
            return [arg for arg in self._args if arg in keys]

    def __repr__(self) -> str:
        mapping = {key: self._storage[key] for key in self._keys()}
        return f"{self.__class__.__name__}({mapping})"


class CustomKeysView(CustomMappingView, KeysView):
    pass


class CustomValuesView(CustomMappingView, ValuesView):
    def __iter__(self) -> Iterable:
        for key in self._keys():
            yield self._storage[key]


class CustomItemsView(CustomMappingView, ItemsView):
    def __iter__(self) -> Iterable[Tuple[str, Any]]:
        for key in self._keys():
            yield key, self._storage[key]


class VirtualAttributesNodeStorage(VirtualAttributesBaseStorage, NodeStorage):
    pass


class VirtualAttributesEdgeStorage(VirtualAttributesBaseStorage, EdgeStorage):
    pass
