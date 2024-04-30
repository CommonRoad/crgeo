from __future__ import annotations
from collections import OrderedDict
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
import numpy as np
from typing import Dict, List, Optional, Set
import gymnasium.spaces
import logging
from commonroad_geometric.common.torch_utils.helpers import flatten_data
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

logger = logging.getLogger(__name__)


class FlattenedGraphObserver(BaseObserver):
    """
    Flattens a CommonRoadData graph object into a dictionary of fixed-sized, padded numpy arrays.
    """

    def __init__(
        self,
        data_padding_size: Optional[int] = None,
        global_features_include: Optional[List[str]] = None
    ) -> None:
        self.data_padding_size = data_padding_size
        self.global_features_include = list(global_features_include) if global_features_include is not None else None
        self._total_step_counter = 0
        self._last: Optional[Dict[str, np.ndarray]] = None
        super().__init__()

    def setup(self, dummy_data: CommonRoadData) -> gymnasium.Space:
        observation_space_dict: OrderedDict[str, gymnasium.spaces.Box] = OrderedDict()
        global_features_include = set() if self.global_features_include is None else set(self.global_features_include)

        ignore_keys = set(dummy_data._global_store.keys()) - global_features_include  # type: ignore
        ignore_keys.update({'vehicle-batch', 'lanelet-batch'})
        if self.data_padding_size is None:
            self.data_padding_size = 500 + 5 * (dummy_data.num_edges + dummy_data.num_nodes)  # TODO
        flattened_data = flatten_data(
            dummy_data,
            self.data_padding_size,
            ignore_keys=ignore_keys
        )

        for k, v in flattened_data.items():
            if k.endswith('-x') or k.endswith('-edge_attr'):
                continue
            dtype = v.detach().numpy().dtype
            if np.dtype(dtype).kind == 'b':
                observation_space_dict[k] = gymnasium.spaces.Box(
                    np.zeros(list(v.shape), dtype='bool'), np.ones(list(v.shape), dtype='bool'), list(v.shape), dtype=dtype
                )
            else:
                observation_space_dict[k] = gymnasium.spaces.Box(
                    -np.inf, np.inf, list(v.shape), dtype=dtype
                )

        observation_space = gymnasium.spaces.Dict(observation_space_dict)
        self._observation_space = observation_space
        return observation_space

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        assert self.data_padding_size is not None

        obs_dict = flatten_data(
            data,
            padding=self.data_padding_size,
            validate=self._total_step_counter == 0,
            ignore_keys={'vehicle-batch', 'lanelet-batch'},
        )
        # TODO do we need to convert to numpy?
        obs_dict = {k: v.detach().squeeze(-1).numpy() if v.ndim > 2 else v.detach().numpy()
                    for k, v in obs_dict.items() if k in self._observation_space.spaces}  # TODO
        self._total_step_counter += 1

        if self.global_features_include is not None:
            for key in self.global_features_include:
                if key not in obs_dict:
                    logger.warning(f"key not in obs_dict: {key}")
                    obs_dict[key] = self._last[key]

        self._last = obs_dict

        for k, v in obs_dict.items():
            assert v.shape == self._observation_space.spaces[
                k].shape, f"unexpected shape of feature {k} ({v.shape} != {self._observation_space.spaces[k].shape})"
        for k in self._observation_space.spaces:
            assert k in obs_dict, f"{k} missing from obs_dict"

        return obs_dict
