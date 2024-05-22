from pyglet import gl

from commonroad_geometric.rendering.viewer.pyglet.attr.attr import Attr


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)

    def disable(self):
        gl.glDisable(gl.GL_LINE_STIPPLE)
