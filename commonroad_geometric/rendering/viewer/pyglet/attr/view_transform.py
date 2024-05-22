import numpy as np
from pyglet import gl

from commonroad_geometric.rendering.viewer.pyglet.attr.attr import Attr


class ViewTransform(Attr):
    """View is centered at center_position"""

    def __init__(
        self,
        center_position: tuple[float, float],
        rotation: float,  # [radians]
        viewport_size: tuple[float, float],
        scale: tuple[float, float],
    ):
        self.center_position = center_position
        self.rotation = rotation
        self.viewport_size = viewport_size
        self.scale = scale

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.viewport_size[0] * 0.5, self.viewport_size[1] * 0.5, 0)
        gl.glRotatef(np.rad2deg(self.rotation), 0.0, 0.0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1.0)
        gl.glTranslatef(-self.center_position[0], -self.center_position[1], 0)

    def disable(self):
        gl.glPopMatrix()
