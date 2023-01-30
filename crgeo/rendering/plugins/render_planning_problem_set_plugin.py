from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from commonroad.geometry.shape import Rectangle
from crgeo.common.geometry.helpers import cut_polyline
from crgeo.rendering.base_renderer_plugin import BaseRendererPlugin
from crgeo.rendering.defaults import ColorTheme
from crgeo.rendering.types import RenderParams
from crgeo.rendering.color_utils import T_ColorTuple
from crgeo.rendering.viewer.viewer_2d import Viewer2D


@dataclass
class RenderPlanningProblemStyle():
    initial_state_linewidth: Optional[float] = 0.1
    initial_state_color: Optional[T_ColorTuple] = None
    render_trajectory: bool = True
    render_start_waypoints: bool = True
    render_goal_waypoints: bool = True
    render_look_ahead_point: bool = True
    trajectory_linewidth: float = 0.6
    trajectory_completed_color: T_ColorTuple = (0.1, 0.1, 0.4, 1.0)
    trajectory_uncompleted_color: T_ColorTuple = (1.0, 1.0, 1.0, 1.0)
    goal_state_linewidth: Optional[float] = 0.1
    goal_state_color: Optional[T_ColorTuple] = None
    circles: bool = True

class RenderPlanningProblemSetPlugin(BaseRendererPlugin):
    def __init__(
        self,
        options: Optional[RenderPlanningProblemStyle] = None,
        **kwargs
    ) -> None:
        self.options = options if options is not None else RenderPlanningProblemStyle(**kwargs)
        
    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.ego_route is None:
            return

        planning_problem_set = params.ego_vehicle.ego_route.planning_problem_set
        goal_state_color = params.render_kwargs.pop('goal_state_color', self.options.goal_state_color)
        initial_state_color = params.render_kwargs.pop('initial_state_color', self.options.initial_state_color)

        if initial_state_color is None:
            if viewer.theme == ColorTheme.DARK:
                initial_state_color = (0.1, 0.1, 0.4, 1.0)
            else:
                initial_state_color = (0.2, 0.9, 0.2, 1.0)
        
        if goal_state_color is None:
            if viewer.theme == ColorTheme.DARK:
                goal_state_color = (1.0, 1.0, 1.0, 1.0)
            else:
                goal_state_color = (0.2, 0.9, 0.2, 1.0)

        if self.options.render_look_ahead_point and params.ego_vehicle.ego_route.look_ahead_point is not None:
            viewer.draw_circle(
                origin=params.ego_vehicle.ego_route.look_ahead_point,
                radius=0.7,
                filled=True,
                color=goal_state_color,
                linewidth=self.options.goal_state_linewidth
            )

        for planning_problem in planning_problem_set.planning_problem_dict.values():
            if self.options.render_goal_waypoints:
                draw_rectangle: bool = True
                for goal_state in planning_problem.goal.state_list:
                    if 'position' not in goal_state.attributes:
                        return
                    if isinstance(goal_state.position, Rectangle) and not self.options.circles:
                        length = goal_state.position.length
                        width = goal_state.position.width
                        vertices = np.array([
                            [- 0.5 * length, - 0.5 * width], [- 0.5 * length, + 0.5 * width],
                            [+ 0.5 * length, + 0.5 * width], [+ 0.5 * length, - 0.5 * width],
                            [- 0.5 * length, - 0.5 * width]
                        ])
                        viewer.draw_shape(
                            vertices=vertices,
                            position=goal_state.position.center,
                            angle=goal_state.position.orientation,
                            filled=True,
                            color=goal_state_color,
                            linewidth=self.options.goal_state_linewidth
                        )
                    else:
                        viewer.draw_circle(
                            origin=goal_state.position.center,
                            radius=goal_state.position.radius if hasattr(goal_state.position, 'radius') else 0.76,
                            filled=True,
                            color=goal_state_color,
                            linewidth=self.options.goal_state_linewidth
                        )
                        draw_rectangle = False
                
                if self.options.render_start_waypoints:
                    if draw_rectangle and not self.options.circles:
                        viewer.draw_shape(
                            vertices=vertices,
                            position=planning_problem.initial_state.position,
                            angle=planning_problem.initial_state.orientation,
                            filled=True,
                            color=initial_state_color,
                            linewidth=self.options.initial_state_linewidth
                        )
                    else:
                        viewer.draw_circle(
                            origin=planning_problem.initial_state.position,
                            radius=planning_problem.initial_state.position.radius if hasattr(planning_problem.initial_state.position, 'radius') else 0.75,
                            filled=True,
                            color=initial_state_color,
                            linewidth=self.options.initial_state_linewidth
                        )

            if self.options.render_trajectory:
                planned_ego_trajectory = params.ego_vehicle.ego_route.planning_problem_path
                if planned_ego_trajectory is not None:
                    trajectory_polyline = params.ego_vehicle.ego_route.planning_problem_path_polyline
                    position = params.ego_vehicle.state.position
                    arclength_to_position = trajectory_polyline.get_projected_arclength(position)
                    cut_polylines = cut_polyline(line=planned_ego_trajectory,
                                                distance=arclength_to_position)

                    viewer.draw_polyline(
                        v=cut_polylines[0],
                        linewidth=self.options.trajectory_linewidth,
                        color=self.options.trajectory_completed_color
                    )

                    if len(cut_polylines) > 1:
                        viewer.draw_polyline(
                            v=cut_polylines[1],
                            linewidth=self.options.trajectory_linewidth,
                            color=self.options.trajectory_uncompleted_color
                        )
