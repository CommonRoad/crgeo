from __future__ import annotations

import numpy as np

from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewer2D


class UserQuitInterrupt(KeyboardInterrupt):
    pass


class UserResetInterrupt(KeyboardInterrupt):
    pass


class UserAdvanceScenarioInterrupt(KeyboardInterrupt):
    pass


def get_keyboard_action(gl_viewer: GLViewer2D) -> np.ndarray:
    from pyglet.window import key

    if gl_viewer is None:
        return np.array([0.0, 0.0], dtype=np.float64)
    keys = gl_viewer.keys

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

    action = np.array([lateral_action, longitudinal_action], dtype=np.float64)
    return action
