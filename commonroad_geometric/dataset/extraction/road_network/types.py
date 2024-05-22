import enum
from enum import IntEnum
from typing import Any, Protocol

from commonroad.scenario.lanelet import LaneletNetwork
from networkx import DiGraph

from commonroad_geometric.rendering.color.color import Color


class GraphConversionStep(Protocol):
    def __call__(self, graph: DiGraph, lanelet_network: LaneletNetwork, *args: Any, **kwargs: Any) -> DiGraph: ...


@enum.unique
class LaneletNodeType(IntEnum):
    LANELET_BEGINNING = 0
    LANELET = 1
    LANELET_END = 2
    CONFLICTING = 3
    INTERSECTION = 4


@enum.unique
class LaneletEdgeType(IntEnum):
    PREDECESSOR = 0
    SUCCESSOR = 1
    OPPOSITE_LEFT = 2
    OPPOSITE_RIGHT = 3
    ADJACENT_LEFT = 4
    ADJACENT_RIGHT = 5
    DIAGONAL_LEFT = 6
    DIAGONAL_RIGHT = 7
    DIVERGING = 8
    MERGING = 9
    CONFLICTING = 10
    CONFLICT_LINK = 11


TrafficFlowEdgeConnections = {
    # LaneletEdgeType.PREDECESSOR,
    LaneletEdgeType.SUCCESSOR,
    LaneletEdgeType.ADJACENT_LEFT,
    LaneletEdgeType.ADJACENT_RIGHT,
    # LaneletEdgeType.CONFLICTING,
    # LaneletEdgeType.CONFLICT_LINK
}

LaneletEdgeTypeColorMap = {
    LaneletEdgeType.PREDECESSOR: Color((0.7, 0.7, 0.7)),
    LaneletEdgeType.SUCCESSOR: Color((0, 0, 0)),
    LaneletEdgeType.DIVERGING: Color((0 / 255, 119 / 255, 182 / 255)),
    LaneletEdgeType.MERGING: Color((114 / 255, 9 / 255, 183 / 255)),
    LaneletEdgeType.ADJACENT_LEFT: Color((7 / 255, 42 / 255, 200 / 255)),
    LaneletEdgeType.ADJACENT_RIGHT: Color((30 / 255, 150 / 255, 252 / 255)),
    LaneletEdgeType.OPPOSITE_LEFT: Color((0.4, 0.4, 0.9)),
    LaneletEdgeType.OPPOSITE_RIGHT: Color((0.4, 0.4, 0.9)),
    LaneletEdgeType.DIAGONAL_LEFT: Color((0.0, 0.9, 0.0)),
    LaneletEdgeType.DIAGONAL_RIGHT: Color((0.0, 0.9, 0.0)),
    LaneletEdgeType.CONFLICTING: Color((247 / 255, 37 / 255, 133 / 255)),
    LaneletEdgeType.CONFLICT_LINK: Color((156 / 255, 122 / 255, 151 / 255)),
}

LaneletNodeTypeColorMap = {
    LaneletNodeType.LANELET: Color((0.7, 0.7, 0.7)),
    LaneletNodeType.LANELET_END: Color((0.7, 0.7, 0.7)),
    LaneletNodeType.LANELET_BEGINNING: Color((0.0, 0.0, 0.0)),
    LaneletNodeType.CONFLICTING: Color((0.7, 0.7, 0.7)),
    LaneletNodeType.INTERSECTION: Color((0.7, 0.7, 0.7)),
}
