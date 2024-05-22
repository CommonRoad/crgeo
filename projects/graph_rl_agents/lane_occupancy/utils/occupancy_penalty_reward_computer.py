from typing import Optional
import logging
import torch
import numpy as np
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


logger = logging.getLogger(__name__)


class OccupancyPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        penalty: float = -0.04,
        discount_factor: float = 0.95,
        time_start: float = 0.0,
        time_cutoff: float = 1.5,
        probability_cutoff: float = 0.0,
        use_max: bool = True,
    ) -> None:
        self.penalty = penalty
        self.discount_factor = discount_factor
        self.time_start = time_start
        self.time_cutoff = time_cutoff
        self.probability_cutoff = probability_cutoff
        self._time_cutoff_mask: Optional[int] = None
        self._time_start_mask: Optional[int] = None
        self.use_max = use_max
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        if self._time_cutoff_mask is None:
            self._time_cutoff_mask = int(self.time_cutoff // simulation.dt)
        if self._time_start_mask is None:
            self._time_start_mask = int(self.time_start // simulation.dt)

        try:
            pred = data.ego_shape_occupancy_predictions[self._time_start_mask:self._time_cutoff_mask]
        except AttributeError:
            logger.warning(
                "Data instance does not contain ego shape occupancy predictions. OccupancyEncodingPostProcessor likely not successfully applied. Setting zero reward")
            return 0.0

        discounts = torch.cumprod(
            torch.full(
                (pred.shape[0], ),
                fill_value=self.discount_factor,
                device=data.device,
                dtype=torch.float32),
            0
        )

        mask = pred >= self.probability_cutoff
        if not mask.any():
            return 0.0

        discounted_preds = (pred * discounts)[mask]
        if self.use_max:
            agg_p = discounted_preds.max().item()
        else:
            agg_p = discounted_preds.sum().item()
        penalty = self.penalty * agg_p

        return penalty
