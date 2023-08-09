import colorsys
from typing import Union, Tuple

T_ColorTuple3 = Tuple[float, float, float]
T_ColorTuple4 = Tuple[float, float, float, float]
T_ColorTuple = Union[T_ColorTuple3, T_ColorTuple4]

RawColor = Union[
    str,
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    Tuple[float, float, float],
    Tuple[float, float, float, float],
]


def normalize_color(c: RawColor) -> T_ColorTuple4:
    if isinstance(c, str):
        assert ((len(c) == 7 or len(c) == 9) and c[0] == "#") or (len(c) == 6 or len(c) == 8)
        if len(c) == 7 or len(c) == 9:
            c = c[1:]  # remove "#" symbol

        r, g, b = int(c[0:2], base=16), int(c[2:4], base=16), int(c[4:6], base=16)
        a = int(c[6:8], base=16) if len(c) == 8 else 255
        return r / 255, g / 255, b / 255, a / 255

    assert isinstance(c, tuple) and (len(c) == 3 or len(c) == 4)
    typ = int if isinstance(c[0], int) else float
    assert len(c) == 3 or (typ is int and (0 <= c[3] <= 255)) or (typ is float and (0.0 <= c[3] <= 1.0))

    if typ is int:
        if len(c) == 3:
            return c[0] / 255, c[1] / 255, c[2] / 255, 1.0
        else:
            return c[0] / 255, c[1] / 255, c[2] / 255, c[3] / 255
    else:
        if len(c) == 3:
            return c[0], c[1], c[2], 1.0
        else:
            return c


def adjust_color_lightness(c: T_ColorTuple4, lightness_factor: float) -> T_ColorTuple4:
    h, l, s = colorsys.rgb_to_hls(*c[:3])
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(l * lightness_factor, 1.0)), s)
    return r, g, b, c[3]



def rgb_to_cmyk(r,g,b):
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0.0, 0.0, 0.0, 1.0

    # rgb [0,1] -> cmy [0,1]
    c = 1 - r 
    m = 1 - g 
    y = 1 - b 

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) 
    m = (m - min_cmy) 
    y = (y - min_cmy) 
    k = min_cmy

    return c, m, y, k

def cmyk_to_rgb(c,m,y,k):
    """
    """
    r = 1.0-(c+k)
    g = 1.0-(m+k)
    b = 1.0-(y+k)
    return r,g,b
