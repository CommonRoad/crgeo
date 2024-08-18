import logging
from typing import List, Optional
from pathlib import Path
import sys
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from projects.geometric_models.drivable_area.models.scenario_drivable_area_model import ScenarioDrivableAreaModel, plot_all_vehicles
from projects.geometric_models.drivable_area.models.scenario_temporal_drivable_area_model import ScenarioTemporalDrivableAreaModel
from projects.geometric_models.drivable_area.project import save_compute_occupancy_vectorized
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EncodingPostProcessor(BaseDataPostprocessor):
    """
    Wrapper for obtaining encodings from pretrained drivable area
    representation model.
    """

    def __init__(
        self,
        model_filepath: Path,
        reload_freq: Optional[int] = None,
        enable_decoding: bool = False
    ) -> None:

        self._model_filepath = model_filepath
        self._model = self._load_model()
        self._reload_freq = reload_freq
        self._enable_decoding = enable_decoding
        self._call_count = 0

        self._show_plot = False and sys.gettrace() is not None
        if self._show_plot:
            plt.ion()
            self.fig, self.axs = plt.subplots(5, 2, figsize=(10, 20))  # 4 types of data, each with actual and predicted


    def _load_model(self) -> ScenarioDrivableAreaModel:
        model = ScenarioDrivableAreaModel.load(
            self._model_filepath,
            device='cpu',
            retries=0,
            from_torch=False
        )
        model.eval()
        return model

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        assert simulation is not None

        self._call_count += 1
        if self._reload_freq is not None and self._call_count % self._reload_freq == 0:
            self._model = self._load_model()

        for data in samples:
            if isinstance(self._model, ScenarioTemporalDrivableAreaModel):
                encoding = self._model.encode_and_contextualize(
                    data_obs=data,
                    data_cutoff=data[data.num_graphs - 1],
                    always_present_ids=data[data.num_graphs - 1].v.id[data[data.num_graphs - 1].v.is_ego_mask]
                )
            else:
                encoding = self._model.encoder.forward(data)


            data.encoding = encoding
            if self._enable_decoding:
                predictions = self._model.decode(data, encoding)
                data.prediction = predictions

                # if self._show_plot and self._call_count % 20 == 0:
                #     save_compute_occupancy_vectorized(data)
                #     plot_all_vehicles(data, predictions, encodings=encoding, fig=self.fig, axs=self.axs)

                #     # Redraw the plot
                #     plt.draw()
                #     # Pause to update the plot
                #     plt.pause(0.1)
        return samples
