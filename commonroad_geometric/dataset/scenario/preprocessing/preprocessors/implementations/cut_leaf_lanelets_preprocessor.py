from __future__ import annotations

import logging

import numpy as np

from commonroad_geometric.common.geometry.helpers import resample_polyline
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor

logger = logging.getLogger(__name__)


class CutLeafLaneletsPreprocessor(ScenarioPreprocessor):
    def __init__(
        self, 
        cutoff: float = 25.0
    ) -> None:
        self.cutoff = cutoff
        super(CutLeafLaneletsPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        for lanelet in scenario_bundle.preprocessed_scenario.lanelet_network.lanelets:
            if lanelet.successor and lanelet.predecessor:
                continue
             
            resample_interval = max(30, int(lanelet.distance[-1]))
            lanelet._center_vertices = np.array(
                resample_polyline(lanelet.center_vertices, resample_interval)
            )
            lanelet._left_vertices = np.array(
                resample_polyline(lanelet.left_vertices, resample_interval)
            )
            lanelet._right_vertices = np.array(
                resample_polyline(lanelet.right_vertices, resample_interval)
            )

            lanelet._distance = None
            lanelet._inner_distance = None
            lanelet.distance
            lanelet.inner_distance

            if not lanelet.successor:
                # exit lanelet
                index_filter = lanelet.distance < self.cutoff
            elif not lanelet.predecessor:
                # enter lanelet
                index_filter = lanelet.distance >= (lanelet.distance[-1] - self.cutoff)

            # TODO: this does not yield clean cutoffs 

            intervals = lanelet.center_vertices.shape[0]
            lanelet._center_vertices = np.array(
                resample_polyline(lanelet.center_vertices[index_filter], intervals)
            )
            lanelet._left_vertices = np.array(
                resample_polyline(lanelet.left_vertices[index_filter], intervals)
            )
            lanelet._right_vertices = np.array(
                resample_polyline(lanelet.right_vertices[index_filter], intervals)
            )
            # hack for recomputing
            lanelet._distance = None
            lanelet._inner_distance = None
            lanelet.distance
            lanelet.inner_distance

        return [scenario_bundle]
