from typing import Dict, Optional

from crgeo.common.geometry.helpers import relative_orientation
from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, LFeatureParams
from crgeo.dataset.extraction.traffic.feature_computers.implementations.types import L_Feature
from crgeo.simulation.base_simulation import BaseSimulation


class LaneletGeometryFeatureComputer(BaseFeatureComputer[LFeatureParams]):


    def __init__(
        self,
    ) -> None:
        self._feature_cache: Dict[int, FeatureDict] = {}
        self._last_scenario_id: Optional[str] = None
        super().__init__()

    def __call__(
        self,
        params: LFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        source_id = params.lanelet.lanelet_id

        if source_id in self._feature_cache:
            return self._feature_cache[source_id]

        source_lanelet_path = simulation.get_lanelet_center_polyline(source_id)

        start_curvature = source_lanelet_path.get_curvature(0.0)
        end_curvature = source_lanelet_path.get_curvature(source_lanelet_path.length)
        direction_change = relative_orientation(
             source_lanelet_path.get_direction(0.0),
             source_lanelet_path.get_direction(source_lanelet_path.length)
        )
        features: FeatureDict = {
            L_Feature.StartCurvature.value: start_curvature,
            L_Feature.EndCurvature.value: end_curvature,
            L_Feature.DirectionChange.value: direction_change,
        }

        self._feature_cache[source_id] = features

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        scenario_id = str(simulation.current_scenario.scenario_id)
        if scenario_id != self._last_scenario_id:
            self._feature_cache = {}
        self._last_scenario_id = scenario_id
