import pyglet

from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom


class Image(GlGeom):
    def __init__(self, fname, width, height, color):
        super(Image, self).__init__(color)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render_geom(self):
        self.img.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)
