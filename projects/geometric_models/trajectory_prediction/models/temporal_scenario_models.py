from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

import torch
from torch import BoolTensor, Tensor

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.common.types import Unlimited
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletNetworkPlugin
from projects.geometric_models.trajectory_prediction.models.decoder.vehicle_model import KinematicSingleTrackVehicleStates, RelativePositionAndOrientationVehicleModel, \
    RelativePositionAndOrientationVehicleStates
from projects.geometric_models.trajectory_prediction.models.decoder.vehicle_trajectory_prediction import VehicleTrajectoryPredictionCVAEDecoder, VehicleTrajectoryPredictionDecoder
from projects.geometric_models.trajectory_prediction.models.encoder.scenario_encoder import ScenarioEncoderModel
from projects.geometric_models.trajectory_prediction.utils.visualization.render_plugins import RenderTrajectoryPredictionPlugin

T_Prediction = TypeVar("T_Prediction")


class MultiplePredictionWrapper(BaseGeometric, Generic[T_Prediction]):

    def __init__(self, cfg: Config, module: BaseGeometric):
        super().__init__(cfg)
        self.wrapped_module = module

    def _build(
        self,
        batch: CommonRoadData,
        trial=None
    ) -> None:
        self.wrapped_module.build(data=batch)
        self.max_time_steps_temporal_edge = self.cfg.traffic.temporal.max_time_steps_temporal_edge
        if self.max_time_steps_temporal_edge == "unlimited":
            self.max_time_steps_temporal_edge = Unlimited

    def _time_slices(self, data: CommonRoadDataTemporal) -> Iterable[Tuple[int, List[CommonRoadData]]]:
        T_obs = self.cfg.traffic.temporal.steps_observe
        T_pred = self.cfg.traffic.temporal.steps_predict
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
        if not isinstance(data, CommonRoadDataTemporal):
            data = data[0]  # TODO avoid batching
        T_obs = self.cfg.traffic.temporal.steps_observe
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

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRenderPlugin]]:
        return [
            RenderLaneletNetworkPlugin(
                lanelet_color=Color((0.65, 0.65, 0.65, 0.8)),
                lanelet_linewidth=0.2
            ),
            # RenderObstaclesPlugin(RenderObstaclesStyle(
            #     render_trail=True,
            #     trail_interval=1
            # )),
            # RenderTrafficGraphPlugin(),
            RenderTrajectoryPredictionPlugin()
        ]

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


@dataclass(frozen=True)
class TrajectoryPrediction(Generic[T_Prediction]):
    rel_time_slice_obs: slice
    rel_time_slice_pred: slice
    data_window_list: List[CommonRoadData]
    vehicle_ids_tensor: Tensor
    vehicle_ids_pred: Tensor
    prediction: T_Prediction


class TemporalTrajectoryPredictionModel(BaseGeometric):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def _build(
        self,
        batch: CommonRoadData,
        trial=None
    ) -> None:
        self.scenario_encoder = ScenarioEncoderModel(cfg=self.cfg)

        # vehicle_constraints = self.cfg.trajectory_prediction.kst_vehicle_constraints
        # self.vehicle_model = KinematicSingleTrackVehicleModel(
        #     velocity_bounds=(vehicle_constraints.velocity_min, vehicle_constraints.velocity_max),
        #     acceleration_bounds=(vehicle_constraints.acceleration_min, vehicle_constraints.acceleration_max),
        #     steering_angle_bound=vehicle_constraints.steering_angle_bound,
        # )
        vehicle_constraints = self.cfg.trajectory_prediction.rpo_vehicle_constraint
        self.vehicle_model = RelativePositionAndOrientationVehicleModel(
            max_velocity=vehicle_constraints.max_velocity,
            max_orientation_delta=vehicle_constraints.max_orientation_delta,
        )

        # assert self.cfg.trajectory_prediction.decoder_type == "CVAE"
        if self.cfg.trajectory_prediction.decoder_type == "CVAE":
            self.trajectory_decoder: VehicleTrajectoryPredictionCVAEDecoder[RelativePositionAndOrientationVehicleModel] = VehicleTrajectoryPredictionCVAEDecoder(
                cfg=self.cfg,
                vehicle_model=self.vehicle_model,
                num_predict_time_steps=self.cfg.traffic.temporal.steps_predict,
                device=batch.device
            )
        else:
            self.trajectory_decoder: VehicleTrajectoryPredictionDecoder[RelativePositionAndOrientationVehicleModel] = VehicleTrajectoryPredictionDecoder(
                cfg=self.cfg,
                vehicle_model=self.vehicle_model,
                num_predict_time_steps=self.cfg.traffic.temporal.steps_predict,
                # device=batch.device
            )

    def _check_vehicle_id_order_consistent_across_time_steps(
        self,
        data: CommonRoadDataTemporal,
        vehicle_mask: BoolTensor,
    ) -> Tuple[Tensor, BoolTensor]:
        T_obs = self.cfg.traffic.temporal.steps_observe
        vehicle_x_slice = data._slice_dict["vehicle"]["x"]
        assert len(vehicle_x_slice) == T_obs + 1 and \
               vehicle_x_slice[0] == 0 and vehicle_x_slice[T_obs] == vehicle_mask.size(0)

        vehicle_mask_0 = vehicle_mask.detach().clone()
        vehicle_mask_0[vehicle_x_slice[1]:] = False
        vehicle_ids_0 = data.vehicle.id[vehicle_mask_0]
        del vehicle_mask_0

        vehicle_mask_t = vehicle_mask.detach().clone()
        for t in range(1, T_obs):
            if t > 1:
                vehicle_mask_t[:] = vehicle_mask
            vehicle_mask_t[:vehicle_x_slice[t]] = False
            vehicle_mask_t[vehicle_x_slice[t + 1]:] = False
            assert torch.all(data.vehicle.id[vehicle_mask_t] == vehicle_ids_0)
        vehicle_mask_last_obs = vehicle_mask_t

        return vehicle_ids_0, vehicle_mask_last_obs

    def forward(
        self,
        data_window_list: List[CommonRoadData],
        data_window_obs: CommonRoadDataTemporal,
    ) -> TrajectoryPrediction:
        T_obs = self.cfg.traffic.temporal.steps_observe
        T_pred = self.cfg.traffic.temporal.steps_predict
        device = self.device

        # encode observed scenario time steps
        vehicle_x = self.scenario_encoder(data_window_obs)
        # TODO decode drivable area from vehicle_x, add drivable area loss to overall loss

        # mask out vehicles which do not exist in all time steps
        vehicle_ids: set[int] = set(data_window_list[0].vehicle.id.view(-1).tolist())
        for data_w in data_window_list[1:]:
            ids_w = set(data_w.vehicle.id.view(-1).tolist())
            vehicle_ids.intersection_update(ids_w)
        N_veh = len(vehicle_ids)
        vehicle_ids_tensor = torch.tensor(list(vehicle_ids), dtype=torch.int, device=device)
        vehicle_mask: BoolTensor = torch.any(data_window_obs.vehicle.id == vehicle_ids_tensor.unsqueeze(0), dim=1)
        # assert vehicle_mask.sum() == len(vehicle_ids) * T_obs

        vehicle_ids_pred, vehicle_mask_last_obs = self._check_vehicle_id_order_consistent_across_time_steps(
            data=data_window_obs,
            vehicle_mask=vehicle_mask
        )
        # vehicle_mask_last_obs masks out all vehicles except the ones in the last observed time step (T_obs - 1)

        vehicle_states_last_obs = KinematicSingleTrackVehicleStates(
            position=data_window_obs.vehicle.pos[vehicle_mask_last_obs],
            velocity_long=data_window_obs.vehicle.velocity[vehicle_mask_last_obs][:, 0].unsqueeze(-1),
            acceleration_long=data_window_obs.vehicle.acceleration[vehicle_mask_last_obs][:, 0].unsqueeze(-1),
            orientation=data_window_obs.vehicle.orientation[vehicle_mask_last_obs],
            length_wheel_base=data_window_obs.vehicle.length[vehicle_mask_last_obs],
        )

        vehicle_x_obs = vehicle_x[vehicle_mask]
        D_vehicle_x = vehicle_x_obs.size(-1)
        assert_size(vehicle_x_obs, (T_obs * N_veh, D_vehicle_x))
        # vehicle_x_obs contains vehicle feature vectors, grouped by time step and in the same order for each time
        # step.
        # For trajectory prediction we need the vehicle feature vectors grouped by vehicle id and within each group
        # ordered by increasing time step.
        vehicle_x_obs_by_vehicle = torch.empty(
            (N_veh, T_obs, D_vehicle_x),
            dtype=vehicle_x_obs.dtype,
            device=vehicle_x_obs.device,
        )
        # TODO replace for-loop with "advanced indexing"
        for v in range(N_veh):
            vehicle_x_obs_by_vehicle[v] = vehicle_x_obs[v::N_veh]

        trajectory_targets: Optional[Tensor] = None
        if self.cfg.trajectory_prediction.decoder_type == "CVAE" and self.training:
            # build trajectory targets for CVAE
            trajectory_targets = torch.empty(
                (N_veh, T_pred, self.vehicle_model.num_state_dims),
                dtype=torch.float32, device=device,
            )
            for t_pred in range(T_pred):
                data_t = data_window_list[T_obs + t_pred]
                vehicle_mask_t: BoolTensor = torch.any(data_t.vehicle.id == vehicle_ids_tensor.unsqueeze(0), dim=1)
                # trajectory_targets[:, t_pred] = KinematicSingleTrackVehicleStates(
                #     position=data_t.vehicle.pos[vehicle_mask_t],
                #     velocity_long=data_t.vehicle.velocity[vehicle_mask_t][:, 0].unsqueeze(-1),
                #     acceleration_long=data_t.vehicle.acceleration[vehicle_mask_t][:, 0].unsqueeze(-1),
                #     orientation=data_t.vehicle.orientation[vehicle_mask_t],
                #     length_wheel_base=None,
                # ).to_tensor()
                trajectory_targets[:, t_pred] = RelativePositionAndOrientationVehicleStates(
                    position=data_t.vehicle.pos[vehicle_mask_t],
                    orientation=data_t.vehicle.orientation[vehicle_mask_t],
                ).to_tensor()

        prediction = self.trajectory_decoder(
            vehicle_x=vehicle_x_obs_by_vehicle,
            vehicle_states_last_obs=vehicle_states_last_obs,
            dt=self.cfg.trajectory_prediction.delta_time,
            trajectory_targets=trajectory_targets,
        )
        return TrajectoryPrediction(
            rel_time_slice_obs=slice(0, T_obs),
            rel_time_slice_pred=slice(T_obs, T_obs + T_pred),
            data_window_list=data_window_list,
            vehicle_ids_tensor=vehicle_ids_tensor,
            vehicle_ids_pred=vehicle_ids_pred,
            prediction=prediction,
        )

    def compute_loss(
        self,
        prediction: TrajectoryPrediction,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_vehicle_states = []
        for t in range(prediction.rel_time_slice_pred.stop - prediction.rel_time_slice_pred.start):
            data_t = prediction.data_window_list[prediction.rel_time_slice_pred.start + t]
            vehicle_mask = torch.any(data_t.vehicle.id == prediction.vehicle_ids_tensor, dim=1)
            assert torch.all(prediction.vehicle_ids_pred == data_t.vehicle.id[vehicle_mask])

            # vehicle_states_pred = KinematicSingleTrackVehicleStates(
            #     position=data_t.vehicle.pos[vehicle_mask],
            #     velocity_long=data_t.vehicle.velocity[vehicle_mask][:, 0].unsqueeze(-1),
            #     acceleration_long=data_t.vehicle.acceleration[vehicle_mask][:, 0].unsqueeze(-1),
            #     orientation=data_t.vehicle.orientation[vehicle_mask],
            #     length_wheel_base=data_t.vehicle.length[vehicle_mask],
            # )
            vehicle_states_pred = RelativePositionAndOrientationVehicleStates(
                position=data_t.vehicle.pos[vehicle_mask],
                orientation=data_t.vehicle.orientation[vehicle_mask],
            )
            target_vehicle_states.append(vehicle_states_pred)

        return self.trajectory_decoder.compute_loss(
            prediction=prediction.prediction,
            target_vehicle_states=target_vehicle_states,
        )


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
