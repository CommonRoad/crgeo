from __future__ import annotations

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

if TYPE_CHECKING:
    from crgeo.rendering.traffic_scene_renderer import TrafficSceneRenderer


class UserQuitInterrupt(KeyboardInterrupt):
    pass


class UserResetInterrupt(KeyboardInterrupt):
    pass


class UserAdvanceScenarioInterrupt(KeyboardInterrupt):
    pass


def get_keyboard_action(renderer: Optional[TrafficSceneRenderer]) -> np.ndarray:
    from pyglet.window import key
    import numpy as np

    if renderer is None:
        return np.array([0.0, 0.0], dtype=np.float64)
    keys = renderer.keys

    if keys[key.R]:
        raise UserResetInterrupt()
    if keys[key.A]:
        raise UserAdvanceScenarioInterrupt()
    if keys[key.Q]:
        raise UserQuitInterrupt()

    if keys[key.UP]:
        longitudinal_action = 1.0
    elif keys[key.DOWN]:
        longitudinal_action = -1.0
    else:
        longitudinal_action = 0.0
    if keys[key.RIGHT]:
        lateral_action = -1.0
    elif keys[key.LEFT]:
        lateral_action = 1.0
    else:
        lateral_action = 0.0

    action = 1000*np.array([lateral_action, longitudinal_action], dtype=np.float64)
    return action
