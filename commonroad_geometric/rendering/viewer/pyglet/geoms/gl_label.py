import pyglet

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom


class GlLabel(GlGeom):
    def __init__(
        self,
        creator: str,
        text: str,
        color: Color,
        font_name: str = None,
        font_size: float = 4.5,
        bold: bool = False,
        italic: bool = False,
        stretch: bool = False,
        x: int = 0,
        y: int = 0,
        width: float = None,
        height: float = None,
        anchor_x: str = 'center',
        anchor_y: str = 'center',
        align: str = 'left',
        multiline: bool = False,
        dpi: int = None,
    ):
        super().__init__(creator=creator)
        self.text = text
        self.color = color
        self.font_name = font_name
        self.font_size = font_size
        self.bold = bold
        self.italic = italic
        self.stretch = stretch
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.align = align
        self.multiline = multiline
        self.dpi = dpi

    def render_geom(self):
        pyglet.text.Label(
            text=self.text,
            font_name=self.font_name,
            font_size=self.font_size,
            bold=self.bold,
            italic=self.italic,
            stretch=self.stretch,
            color=self.color,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            anchor_x=self.anchor_x,
            anchor_y=self.anchor_y,
            align=self.align,
            multiline=self.multiline,
            dpi=self.dpi
        ).draw()
