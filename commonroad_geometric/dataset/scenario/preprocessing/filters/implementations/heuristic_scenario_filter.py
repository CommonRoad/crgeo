import logging
from dataclasses import dataclass
from functools import cached_property
from sys import maxsize
from typing import Iterable, Optional, Set, Tuple

import numpy as np
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.logging import BraceMessage as __
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter

logger = logging.getLogger(__name__)


@dataclass
class MinimumRequirements:
    num_lanelets: int = 2
    total_lanelet_length: float = 170.0
    max_lanelet_curvature: float = 0.32
    max_segment_curvature: float = 0.03
    max_segment_curvature_per_length: float = 0.005
    min_lanelet_width: float = 2.4
    max_lanelet_width: float = 7.0

    @cached_property
    def lower_bounds(self) -> dict[str, float]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name != "max_lanelet_width"
        }

    @cached_property
    def upper_bounds(self) -> dict[str, float]:
        return dict(max_lanelet_width=self.max_lanelet_width)


@dataclass
class SufficientRequirements:
    num_intersections: int = 1
    min_lanelet_width: float = 3.6
    max_segment_curvature: float = 0.14
    total_lanelet_curvature: float = 8.5
    max_lanelet_curvature: float = 1.95
    lanelet_curvature_variance: float = 0.64
    max_lanelet_curvature_per_length: float = 0.6
    max_parallel_lanes: int = 4
    max_lanelet_width: float = 3.5

    @cached_property
    def lower_bounds(self) -> dict[str, float]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name != "max_lanelet_width"
        }

    @cached_property
    def upper_bounds(self) -> dict[str, float]:
        return dict(max_lanelet_width=self.max_lanelet_width)


class HeuristicOSMScenarioFilter(ScenarioFilter):
    def __init__(
        self,
        minimum_requirements: Optional[MinimumRequirements] = None,
        sufficient_requirements: Optional[SufficientRequirements] = None,
    ) -> None:
        super().__init__()
        self.minimum_requirements = minimum_requirements or MinimumRequirements()
        self.sufficient_requirements = sufficient_requirements or SufficientRequirements()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        satisfies_min_reqs, reason = self._filter_scenario_requirements(
            requirements=self.minimum_requirements,
            metrics=iterate_scenario_metrics(scenario_bundle.preprocessed_scenario)
        )
        if not satisfies_min_reqs:
            logger.debug(__(r"Rejecting {scenario_path=} because it violates minimum requirement {reason}",
                            scenario_path=scenario_bundle.scenario_path,
                            reason=reason))
            return False

        satisfies_sufficient_reqs, reason = self._filter_scenario_requirements(
            requirements=self.sufficient_requirements,
            metrics=iterate_scenario_metrics(scenario_bundle.preprocessed_scenario)
        )
        if not satisfies_sufficient_reqs:
            logger.debug(__(r"Rejecting {scenario_path=} because it violates sufficient requirement {reason}",
                            scenario_path=scenario_bundle.scenario_path,
                            reason=reason))

        return satisfies_sufficient_reqs

    @staticmethod
    def _filter_scenario_requirements(
        requirements: MinimumRequirements | SufficientRequirements,
        metrics: Iterable[tuple[str, float]]
    ) -> tuple[bool, str]:
        for metric, value in metrics:
            lower_bound = requirements.lower_bounds.get(metric)
            if lower_bound is not None:
                if value < lower_bound:
                    return False, f"{metric=}"
            upper_bound = requirements.upper_bounds.get(metric)
            if upper_bound is not None:
                if value > upper_bound:
                    return False, f"{metric=}"
        return True, "accept"


def iterate_scenario_metrics(scenario: Scenario) -> Iterable[Tuple[str, float]]:
    lanelets = scenario.lanelet_network.lanelets
    yield "num_lanelets", len(lanelets)
    yield "num_intersections", len(scenario.lanelet_network.intersections)

    # 2 metric types: per lanelet, per lanelet segment

    yield "total_lanelet_length", sum(
        lanelet.distance[-1]
        for lanelet in lanelets
    )

    lanelet_widths = [
        np.linalg.norm(lanelet.left_vertices - lanelet.right_vertices, axis=-1)
        for lanelet in lanelets
    ]
    yield "max_lanelet_width", max(widths.max() for widths in lanelet_widths)
    yield "min_lanelet_width", min(widths.min() for widths in lanelet_widths)

    lanelet_curvatures = [
        compute_curvature(lanelet.center_vertices)
        for lanelet in scenario.lanelet_network.lanelets
    ]

    yield "max_segment_curvature", max(
        np.abs(rel_angles).max()  # max abs aggregation
        for rel_angles, _ in lanelet_curvatures
    )

    abs_lanelet_curvatures = [
        np.abs(rel_angles).sum()  # absolute sum aggregation
        for rel_angles, _ in lanelet_curvatures
    ]
    yield "total_lanelet_curvature", sum(abs_lanelet_curvatures)

    # in [rad / m]
    yield "max_lanelet_curvature_per_length", max(
        abs_curvature / lanelet.distance[-1]
        for abs_curvature, lanelet in zip(abs_lanelet_curvatures, scenario.lanelet_network.lanelets)
    )

    yield "max_lanelet_curvature", max(abs_lanelet_curvatures)

    yield "lanelet_curvature_variance", np.var(abs_lanelet_curvatures)

    yield "max_segment_curvature_variance", max(
        np.var(rel_angles)  # variance aggregation
        for rel_angles, _ in lanelet_curvatures
    )

    yield "max_lanelet_curvature_per_length", max(
        abs_curvature / lanelet.distance[-1]
        for lanelet, abs_curvature in zip(lanelets, abs_lanelet_curvatures)
    )

    max_curvature_per_segment_length = 0.0
    for rel_angles, segments in lanelet_curvatures:
        segment_lengths = np.linalg.norm(segments, axis=1, keepdims=True)
        segment_lengths_around_angles = np.array([segment_lengths[:-1], segment_lengths[1:]]).sum(axis=0)
        lanelet_max_curvature_per_segment_length = np.max(
            np.absolute(rel_angles) /
            segment_lengths_around_angles) if len(segment_lengths_around_angles) > 0 else 0.0
        max_curvature_per_segment_length = max(
            lanelet_max_curvature_per_segment_length,
            max_curvature_per_segment_length,
        )
    yield "max_segment_curvature_per_length", max_curvature_per_segment_length

    max_parallel_lanes = 0
    visited_lanelets: Set[int] = set()
    for lanelet in lanelets:
        if lanelet.lanelet_id in visited_lanelets:
            continue
        visited_lanelets.add(lanelet.lanelet_id)

        num_lanes = 1
        for dir in ["left", "right"]:
            current = lanelet
            while True:
                next_id = getattr(current, f"adj_{dir}")
                if next_id is None or not getattr(lanelet, f"adj_{dir}_same_direction"):
                    break

                if next_id in visited_lanelets:
                    num_lanes = -maxsize
                    break

                num_lanes += 1
                visited_lanelets.add(next_id)
                current = scenario.lanelet_network.find_lanelet_by_id(next_id)

        max_parallel_lanes = max(max_parallel_lanes, num_lanes)
    yield "max_parallel_lanes", max_parallel_lanes
