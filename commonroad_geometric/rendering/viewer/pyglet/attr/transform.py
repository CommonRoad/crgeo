import numpy as np
from pyglet import gl

from commonroad_geometric.rendering.viewer.pyglet.attr.attr import Attr


class Transform(Attr):
    def __init__(
        self,
        translation: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,  # [radians]
        scale: tuple[float, float] = (1.0, 1.0),
    ):
        self.translation = translation
        self.rotation = rotation
        self.scale = scale

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.translation[0], self.translation[1], 0)  # translate to GL loc ppint
        gl.glRotatef(np.rad2deg(self.rotation), 0.0, 0.0, 1.0)

    def disable(self):
        gl.glPopMatrix()

    def set_translation(self, x: float, y: float):
        self.translation = (x, y)

    def set_rotation(self, rotation: float):
        self.rotation = rotation

    def set_scale(self, x: float, y: float):
        self.scale = (x, y)
