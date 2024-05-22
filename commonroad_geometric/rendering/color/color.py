from __future__ import annotations

from typing import Optional, Tuple, Union

from matplotlib.colors import Normalize, to_rgba

T_ColorName = str
T_ColorTuple3 = tuple[int, int, int]
T_ColorTuple4 = tuple[int, int, int, float]
T_NormalizedColorTuple3 = Tuple[float, float, float]
T_NormalizedColorTuple4 = Tuple[float, float, float, float]
T_RawColor = Union[
    T_ColorName,
    T_ColorTuple3,
    T_ColorTuple4,
    T_NormalizedColorTuple3,
    T_NormalizedColorTuple4
]


class Color:
    def __init__(
        self,
        color: T_RawColor,
        alpha: Optional[float] = None
    ) -> None:
        self.rgba_color: T_NormalizedColorTuple4 = self.to_normalized_rgba(raw_color=color, alpha=alpha)

    @staticmethod
    def to_normalized_rgba(
        raw_color: T_RawColor,
        alpha: Optional[float] = None
    ) -> T_NormalizedColorTuple4:
        norm = Normalize(vmin=0, vmax=255)
        match raw_color:
            case str() as s:
                return to_rgba(c=s, alpha=alpha)
            case int() as r, int() as g, int() as b:
                normalized_rgb = norm((r, g, b))
                return to_rgba(c=normalized_rgb, alpha=alpha)
            case float() as r, float() as g, float() as b:
                return to_rgba(c=(r, g, b), alpha=alpha)
            case int() as r, int() as g, int() as b, float() as a:
                # Explicit non-None alpha takes precedence here
                normalized_rgb = norm((r, g, b))
                return to_rgba(c=normalized_rgb, alpha=alpha or a)
            case float() as r, float() as g, float() as b, float() as a:
                # Explicit non-None alpha takes precedence here
                return to_rgba(c=(r, g, b), alpha=alpha or a)
            case unknown:
                # Unknown color format, try to convert to RGBA anyway
                normalized_unknown = norm(unknown)
                return to_rgba(c=normalized_unknown, alpha=alpha)

    @property
    def red(self):
        return self.rgba_color[0]

    @property
    def green(self):
        return self.rgba_color[1]

    @property
    def blue(self):
        return self.rgba_color[2]

    @property
    def alpha(self):
        return self.rgba_color[3]

    def as_rgba(self) -> T_NormalizedColorTuple4:
        return self.rgba_color

    def __mul__(self, scaling: float) -> Color:
        scaled_color = (
            self.red * scaling,
            self.green * scaling,
            self.blue * scaling,
            self.alpha  # Don't scale alpha channel
        )  # Like this to get correct type hinting
        return Color(color=scaled_color)

    def with_alpha(self, alpha: float) -> Color:
        return Color(
            color=self.rgba_color,
            alpha=alpha
        )

    def __repr__(self) -> str:
        return f"Color(color={self.rgba_color})"

    def __str__(self) -> str:
        return f"RGBA Color: Red={self.red:.2f}, Green={self.green:.2f}, Blue={self.blue:.2f}, Alpha={self.alpha:.2f}"
