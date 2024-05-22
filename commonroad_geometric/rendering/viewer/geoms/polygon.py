from abc import ABC, abstractmethod

from commonroad_geometric.rendering.types import T_Vertices


class FilledPolygon(ABC):

    @property
    @abstractmethod
    def vertices(self) -> T_Vertices:
        ...
