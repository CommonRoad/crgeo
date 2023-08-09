import itertools
from typing import Set, Tuple, Iterable

import numpy as np
from sys import maxsize
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class HeuristicOSMScenarioFilterer(BaseScenarioFilterer):

    # TODO: make thresholds configurable, move to crgeo core

    def __init__(self):
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum requirements for a scenario
        metrics_min_reqs, metrics_suf_reqs = itertools.tee(get_scenario_metrics(scenario), 2)
        accept, _ = self._pre_filter_scenario_minimum_requirements(metrics=metrics_min_reqs)
        if not accept:
            return False

        # at least one of the following conditions has to be met
        accept, _ = self._pre_filter_scenario_sufficient_requirements(metrics=metrics_suf_reqs)
        return accept

    @staticmethod
    def _pre_filter_scenario_minimum_requirements(metrics: Iterable[Tuple[str, float]]) -> Tuple[bool, str]:
        metrics_lower_bound = {
            "num_lanelets": 2,
            "total_lanelet_length": 170.0,
            "max_lanelet_curvature": 0.32,
            "max_segment_curvature": 0.03,
            "max_segment_curvature_per_length": 0.005,
            "min_lanelet_width": 2.4,
        }
        metrics_upper_bound = {
            "max_lanelet_width": 7.0,
        }
        for metric, value in metrics:
            lower_bound = metrics_lower_bound.get(metric)
            if lower_bound is not None:
                if value < lower_bound:
                    return False, f"reject {metric}"
            upper_bound = metrics_upper_bound.get(metric)
            if upper_bound is not None:
                if value > upper_bound:
                    return False, f"reject {metric}"

        return True, "accept no reject"

    @staticmethod
    def _pre_filter_scenario_sufficient_requirements(metrics: Iterable[Tuple[str, float]]) -> Tuple[bool, str]:
        metrics_lower_bound = {
            "num_intersections": 1,
            "min_lanelet_width": 3.6,
            "max_segment_curvature": 0.14,
            "total_lanelet_curvature": 8.5,
            "max_lanelet_curvature": 1.95,
            "lanelet_curvature_variance": 0.64,
            "max_lanelet_curvature_per_length": 0.6,
            "max_parallel_lanes": 4,
        }
        metrics_upper_bound = {
            "max_lanelet_width": 3.5,
        }
        for metric, value in metrics:
            lower_bound = metrics_lower_bound.get(metric)
            if lower_bound is not None:
                if value >= lower_bound:
                    return True, f"accept {metric}"
            upper_bound = metrics_upper_bound.get(metric)
            if upper_bound is not None:
                if value <= upper_bound:
                    return True, f"accept {metric}"

        return False, "reject no accept"


def get_scenario_metrics(scenario: Scenario) -> Iterable[Tuple[str, float]]:
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
        lanelet_max_curvature_per_segment_length = np.max(np.absolute(rel_angles) / segment_lengths_around_angles) if len(segment_lengths_around_angles) > 0 else 0.0
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
