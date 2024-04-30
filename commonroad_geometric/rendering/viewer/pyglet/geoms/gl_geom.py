from abc import abstractmethod
from typing import Optional

from commonroad_geometric.rendering.viewer.geoms.base_geom import BaseGeom
from commonroad_geometric.rendering.viewer.pyglet.attr.attr import Attr


class GlGeom(BaseGeom):
    r"""
    Interface for pyglet specific behavior for the BaseGeom class.
    """

    def __init__(
        self,
        creator: str,
        attrs: Optional[list[Attr]] = None
    ):
        super().__init__(creator=creator)
        self.attrs = attrs or []

    def render(self):
        for attr in self.attrs:
            attr.enable()
        self.render_geom()
        for attr in self.attrs:
            attr.disable()

    @abstractmethod
    def render_geom(self):
        ...
