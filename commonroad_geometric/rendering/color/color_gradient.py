from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_rgba

from commonroad_geometric.rendering.color.color import Color, T_RawColor


class ColorGradient:
    def __init__(
        self,
        colors: list[T_RawColor],
        min_val: int | float = 0,
        max_val: int | float = 1,
        name: str = "ColorGradient"
    ) -> None:
        self.colors = colors
        self.min_val = min_val
        self.max_val = max_val
        self.cmap = LinearSegmentedColormap.from_list(name, colors)

    def __getitem__(self, index: int | float) -> Color:
        color = self.cmap(index / (self.max_val - self.min_val))
        rgba_color = to_rgba(c=color)
        return Color(rgba_color)
