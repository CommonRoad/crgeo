import os
from abc import ABC, abstractmethod
from typing import Tuple

# Checks if rendering is in rendering headless mode, can be activated by 'export PYGLET_HEADLESS=...' in environment
import pyglet

from commonroad_geometric.common.geometry.helpers import TWO_PI
if "PYGLET_HEADLESS" in os.environ:
    pyglet.options["headless"] = True

# Disable error checking for increased performance
pyglet.options['debug_gl'] = False

from pyglet import gl

import numpy as np
import math


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if hasattr(geom, 'set_linewidth'):
        if "linewidth" in attrs:
            geom.set_linewidth(attrs["linewidth"])
        else:
            geom.set_linewidth(0)


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError()

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr(ABC):

    @abstractmethod
    def enable(self) -> None:
        ...

    def disable(self) -> None:
        pass


class Transform(Attr):
    def __init__(
        self,
        translation: Tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,  # [radians]
        scale: Tuple[float, float] = (1.0, 1.0),
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


class ViewTransform(Attr):
    """View is centered at center_position"""

    def __init__(
        self,
        center_position: Tuple[float, float],
        rotation: float,  # [radians]
        viewport_size: Tuple[int, int],
        scale: Tuple[float, float],
    ):
        self.center_position = center_position
        self.rotation = rotation
        self.viewport_size = viewport_size
        self.scale = scale

    def set_viewport_size(self, width: int, height: int) -> None:
        self.viewport_size = (width, height)

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.viewport_size[0] * 0.5, self.viewport_size[1] * 0.5, 0)
        gl.glRotatef(np.rad2deg(self.rotation), 0.0, 0.0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1.0)
        gl.glTranslatef(-self.center_position[0], -self.center_position[1], 0)

    def disable(self):
        gl.glPopMatrix()


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        gl.glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)

    def disable(self):
        gl.glDisable(gl.GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        if self.stroke > 0.0:
            gl.glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        gl.glBegin(gl.GL_POINTS)  # draw point
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            gl.glBegin(gl.GL_QUADS)
        elif len(self.v) > 4:
            gl.glBegin(gl.GL_POLYGON)
        else:
            gl.glBegin(gl.GL_TRIANGLES)
        for p in self.v:
            gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        gl.glEnd()


def make_circle(origin=(0, 0), radius=10, res=30, filled=True, start_angle=0, end_angle=TWO_PI, return_points=False):
    points = []
    for i in range(res + 1):
        ang = start_angle + i * (end_angle - start_angle) / res
        points.append((math.cos(ang) * radius + origin[0], math.sin(ang) * radius + origin[1]))
    if (return_points):
        return points
    else:
        if filled:
            return FilledPolygon(points)
        else:
            return PolyLine(points, True)


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP)
        for p in self.v:
            gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        gl.glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), linewidth=1):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(linewidth)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Label(Geom):
    def __init__(self, label='', font_name=None, font_size=4.5, x=(0.0, 0.0), y=(0.0, 0.0), bold=False,
                 anchor_x='center', anchor_y='center', color=(255, 255, 255, 255), italic=False, height=None, width=None, **_kwargs):
        Geom.__init__(self)
        self.label = label
        self.x = x
        self.y = y
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.font_size = font_size
        self.font_name = font_name
        self.bold = bold
        self.color = color
        self.italic = italic
        self.width = width
        self.height = height

    def render1(self):
        pyglet.text.Label(text=self.label,
                          font_name=self.font_name,
                          font_size=self.font_size,
                          width=self.width,
                          height=self.height,
                          color=self.color,
                          x=self.x,
                          y=self.y,
                          anchor_x='center',
                          anchor_y='center').draw()


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)
