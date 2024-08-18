from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from torch import BoolTensor, Tensor

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.common.types import Unlimited
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric

T_Prediction = TypeVar("T_Prediction")


class MultiplePredictionWrapper(BaseGeometric, Generic[T_Prediction]):

    def __init__(
        self, 
        module: BaseGeometric,
        steps_observe: int,
        steps_predict: int,
        max_time_steps_temporal_edge: Union[str, int] = "unlimited",
        cfg = None
    ):
        super().__init__()
        self.wrapped_module = module
        self.steps_observe = steps_observe
        self.steps_predict = steps_predict
        self.max_time_steps_temporal_edge = max_time_steps_temporal_edge
        if self.max_time_steps_temporal_edge == "unlimited":
            self.max_time_steps_temporal_edge = Unlimited

    def _build(
        self,
        batch: CommonRoadData,
        trial=None
    ) -> None:
        self.wrapped_module.build(data=batch)

    def _time_slices(self, data: CommonRoadDataTemporal) -> Iterable[Tuple[int, List[CommonRoadData]]]:
        T_obs = self.steps_observe
        T_pred = self.steps_predict
        N = data.num_graphs
        t = 0
        data_list: List[CommonRoadData] = data.index_select(list(range(N)))
        while t + T_obs + T_pred <= N:
            yield t, data_list[t:t + T_obs + T_pred]
            t += T_obs

    def forward(
        self,
        data: CommonRoadDataTemporal
    ) -> List[T_Prediction]:
        T_obs = self.steps_observe
        predictions = []
        for t, data_window_list in self._time_slices(data):
            data_window_obs = CommonRoadDataTemporal.from_data_list(
                data_list=data_window_list[:T_obs],
                delta_time=data.delta_time,
            )
            CommonRoadDataTemporal.add_temporal_vehicle_edges_(
                data=data_window_obs,
                max_time_steps_temporal_edge=self.max_time_steps_temporal_edge,
                # feature_computers=[ft_rel_state_vtv]
            )
            predictions.append(self.wrapped_module(data_window_list, data_window_obs))

        assert len(predictions) > 0
        return predictions

    def compute_loss(
        self,
        predictions: List[T_Prediction],
        data: Any,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss_sum_primary: Optional[Tensor] = None
        loss_sum_other: Optional[Tensor] = None
        for pred in predictions:
            primary_loss, other_losses = self.wrapped_module.compute_loss(prediction=pred)
            loss_sum_other = losses_add_(loss_sum_other, other_losses)
            if loss_sum_primary is None:
                loss_sum_primary = primary_loss
            else:
                loss_sum_primary += primary_loss

        loss_avg_primary = loss_sum_primary / len(predictions)
        loss_avg_other = losses_divide_(loss_sum_other, len(predictions))

        return loss_avg_primary, loss_avg_other
    

def losses_add_(losses1: Optional[Dict[str, Tensor]], losses2: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if losses1 is None:
        return losses2
    for k in losses1.keys():
        losses1[k] = losses1[k] + losses2[k]
    return losses1


def losses_divide_(losses: Dict[str, Tensor], x: float) -> Dict[str, Tensor]:
    for k in losses.keys():
        losses[k] = losses[k] / x
    return losses
