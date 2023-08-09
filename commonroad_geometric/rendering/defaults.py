from enum import Enum


class ColorTheme(Enum):
    DARK = 'dark'
    BRIGHT = 'bright'

# Viewer settings
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_FPS = 40
DEFAULT_VIEW_RANGE = 100.0

# Colors
DEFAULT_OBSTACLE_COLOR = (0.1, 0.8, 0.1, 1.0) # TODO: get rid ofthese

# Shapes
DEFAULT_OBSTACLE_LINEWIDTH = 0.5
