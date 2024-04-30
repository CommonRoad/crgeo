from pyglet import gl

from commonroad_geometric.rendering.viewer.pyglet.attr.attr import Attr


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        if self.stroke > 0.0:
            gl.glLineWidth(self.stroke)

    def disable(self):
        pass
