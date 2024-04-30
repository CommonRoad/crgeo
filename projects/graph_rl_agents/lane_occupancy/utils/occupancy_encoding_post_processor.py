import logging
from copy import deepcopy
from typing import List, Optional
from pathlib import Path

import torch

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from projects.geometric_models.lane_occupancy.models.occupancy.occupancy_model import OccupancyModel

logger = logging.getLogger(__name__)


class OccupancyEncodingPostProcessor(BaseDataPostprocessor):
    """
    Wrapper for obtaining encodings from pretrained occupancy-predictive
    representation model.
    """

    def __init__(
        self,
        model_cls: BaseGeometric,
        model_filepath: Path,
        include_path_decodings: bool = True,
        include_ego_vehicle_decodings: bool = True,
        decoding_resolution_route: int = 50,
        decoding_resolution_vehicle: int = 20,
        decoding_time_horizon: int = 60,
        compute_vehicle_integrals: bool = True,
        ego_length_multiplier: float = 1.0,
        reload_freq: Optional[int] = None,
        deepcopy_data: bool = False,
        masking: bool = False
    ) -> None:
        self._model = model_cls.load(
            model_filepath,
            device='cpu',
            retries=0,
            from_torch=True
        )
        self._model.eval()
        self._model_filepath = model_filepath
        self._include_decodings = include_path_decodings
        self._decoding_resolution_route = decoding_resolution_route
        self._decoding_resolution_vehicle = decoding_resolution_vehicle
        self._decoding_time_horizon = decoding_time_horizon
        self._include_ego_vehicle_decodings = include_ego_vehicle_decodings
        self._compute_vehicle_integrals = compute_vehicle_integrals
        self._ego_length_multiplier = ego_length_multiplier
        self._reload_freq = reload_freq
        self._call_count = 0
        self._deepcopy_data = deepcopy_data
        self._masking = masking

    @property
    def path_length(self) -> float:
        return self._model.config.path_length

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        assert simulation is not None

        self._call_count += 1
        if self._reload_freq is not None and self._call_count % self._reload_freq == 0:
            self._model = OccupancyModel.load(self._model_filepath, device='cpu')
            self._model.eval()

        for data in samples:
            if not hasattr(data, 'walks'):
                logger.warning(f"Cannot encode data, equ sequence post processing not applied")
                continue

            encode_data = deepcopy(data) if self._deepcopy_data else data  # hack to avoid interference
            encode_data.walk_velocity = torch.tensor(
                [ego_vehicle.state.velocity],
                device=data.device,
                dtype=torch.float32
            )

            if self._model.config.path_conditioning:
                self._model.preprocess_conditioning(
                    encode_data,
                    encode_data.walks,
                    encode_data.walk_start_length,
                    walk_masks=encode_data.ego_trajectory_sequence_mask.bool() if self._masking else None
                )
                out = self._model.encode(encode_data)
                z_ego_route, z_r, message_intensities = out
                data.path_length = encode_data.path_length
                data.lanelet.occupancy_encodings = z_r
                data.z_ego_route = z_ego_route.squeeze(0)
                data.message_intensities = message_intensities

                if self._include_decodings:
                    data.ego_route_occupancy_predictions = self._model.decoder.forward(
                        domain=self._decoding_resolution_route,
                        z=z_ego_route,
                        lanelet_length=data.path_length,
                        dt=simulation.dt,
                        time_horizon=self._decoding_time_horizon
                    )

                if self._include_ego_vehicle_decodings:
                    assert ego_vehicle is not None
                    d = self._ego_length_multiplier * ego_vehicle.shape.length / data.path_length
                    t_vec = torch.arange(
                        self._decoding_time_horizon,
                        device=data.device,
                        dtype=torch.float32) * simulation.dt
                    pos_t = ego_vehicle.state.velocity / data.path_length * t_vec
                    linspace = torch.linspace(0, 1, self._decoding_resolution_vehicle,
                                              device=data.device, dtype=torch.float32)
                    lower_t = torch.clamp(pos_t - d / 2, 0.0, 1.0)[:, None]
                    upper_t = torch.clamp(pos_t + d / 2, 0.0, 1.0)[:, None]
                    diff_t = upper_t - lower_t
                    mask = (diff_t > 0).squeeze(1)
                    integration_domain = lower_t + linspace[None, :] * (upper_t - lower_t)
                    occ_probs = self._model.decoder.forward(
                        domain=integration_domain,
                        z=z_ego_route,
                        lanelet_length=data.path_length,
                        dt=simulation.dt,
                        time_horizon=self._decoding_time_horizon
                    )[0][mask]
                    if self._compute_vehicle_integrals:
                        ego_shape_occupancy_predictions = torch.trapz(
                            occ_probs,
                            x=integration_domain[mask]
                        ) / diff_t.squeeze(1)[mask]
                    else:
                        ego_shape_occupancy_predictions = torch.max(occ_probs, dim=1)[0]
                    data.ego_shape_pos_t = pos_t
                    data.ego_shape_occupancy_predictions = ego_shape_occupancy_predictions

                del data.vehicle.batch
                del data.lanelet.batch

                pass
            else:
                out = self._model.encode(data)
                z_lanelet = out[0] if isinstance(out, tuple) else out
                data.lanelet.occupancy_encodings = z_lanelet
                if self._include_decodings:
                    data.lanelet.occupancy_predictions = self._model.decoder.forward(
                        domain=self._decoding_resolution_route,
                        z=z_lanelet,
                        lanelet_length=data.l.length,
                        dt=simulation.dt,
                        time_horizon=self._decoding_time_horizon
                    )

        return samples
