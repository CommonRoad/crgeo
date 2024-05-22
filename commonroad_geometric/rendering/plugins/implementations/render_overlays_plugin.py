from dataclasses import dataclass

from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderOverlayPlugin(BaseRenderPlugin):
    def __init__(
        self,
        disable_overlays: bool = False
    ) -> None:
        self._disable_overlays = disable_overlays
        super(RenderOverlayPlugin, self).__init__()

    def disable_overlays(self) -> None:
        self._disable_overlays = True

    def enable_overlays(self) -> None:
        self._disable_overlays = True

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if self._disable_overlays:
            return

        overlays = params.render_kwargs.get('overlays', None)
        if overlays is None:
            return

        lines = []
        for overlay_key, value in overlays.items():
            # TODO: Make less hacky
            lines.append(f"{(overlay_key + ':' if overlay_key else ''):<35}{numpy_prettify(value):<30}")

        text = '\n'.join(lines)
        viewer.draw_label(
            creator=self.__class__.__name__,
            text=text,
            color=(0, 255, 0, 255) if viewer.options.theme == ColorTheme.DARK else (0, 0, 0, 255),
            font_size=10,
            x=viewer.width - 30,
            y=viewer.height - 30,
            anchor_x='right',
            anchor_y='top',
            multiline=True,
            width=350,
            height=600,
            dpi=85,
        )
