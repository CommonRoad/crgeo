from dataclasses import dataclass
from typing import Optional

import numpy as np
from commonroad.geometry.shape import Rectangle

from commonroad_geometric.common.geometry.helpers import cut_polyline
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderPlanningProblemStyle:
    initial_state_linewidth: Optional[float] = 0.1
    initial_state_color: Optional[Color] = None
    render_trajectory: bool = True
    render_start_waypoints: bool = True
    render_goal_waypoints: bool = True
    render_look_ahead_point: bool = True
    trajectory_linewidth: float = 0.6
    trajectory_completed_color: Color = Color((0.1, 0.1, 0.4, 1.0))
    trajectory_uncompleted_color: Color = Color((0.5, 0.5, 0.5, 1.0))
    goal_state_linewidth: Optional[float] = 0.1
    goal_state_color: Optional[Color] = None
    circles: bool = True


class RenderPlanningProblemSetPlugin(BaseRenderPlugin):

    def __init__(
        self,
        options: Optional[RenderPlanningProblemStyle] = None,
        **kwargs
    ) -> None:
        self.options = options if options is not None else RenderPlanningProblemStyle(**kwargs)
        super(RenderPlanningProblemSetPlugin, self).__init__()

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.ego_route is None:
            return

        planning_problem_set = params.ego_vehicle.ego_route.planning_problem_set
        goal_state_color = params.render_kwargs.pop('goal_state_color', self.options.goal_state_color)
        initial_state_color = params.render_kwargs.pop('initial_state_color', self.options.initial_state_color)

        if initial_state_color is None:
            if viewer.options.theme == ColorTheme.DARK:
                initial_state_color = Color((0.1, 0.1, 0.4, 1.0))
            else:
                initial_state_color = Color((0.2, 0.9, 0.2, 1.0))

        if goal_state_color is None:
            if viewer.options.theme == ColorTheme.DARK:
                goal_state_color = Color((1.0, 1.0, 1.0, 1.0))
            else:
                goal_state_color = Color((0.2, 0.9, 0.2, 1.0))

        if self.options.render_look_ahead_point:
            for look_ahead_distance, look_ahead_point in params.ego_vehicle.ego_route.look_ahead_points.items():
                path_heading = params.ego_vehicle.ego_route.planning_problem_path_polyline.get_projected_direction(
                    look_ahead_point
                )
                path_direction = np.array([
                    np.cos(path_heading),
                    np.sin(path_heading)
                ])
                scale = 20.0

                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=look_ahead_point,
                    radius=0.7,
                    fill_color=goal_state_color,
                    border_color=goal_state_color,
                    line_width=self.options.goal_state_linewidth
                )

                arrow_origin = look_ahead_point
                arrow_end = np.array([
                    arrow_origin[0] + path_direction[0] * scale,
                    arrow_origin[1] + path_direction[1] * scale,
                ])

                viewer.draw_arrow_from_to(
                    creator=self.__class__.__name__,
                    start=arrow_origin,
                    end=arrow_end,
                    line_color=goal_state_color,  # Adjust the color as needed
                    line_width=4  # Adjust the line width as needed
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
                        viewer.draw_2d_shape(
                            creator=self.__class__.__name__,
                            vertices=vertices,
                            fill_color=goal_state_color,
                            border_color=goal_state_color,
                            translation=goal_state.position.center,
                            rotation=goal_state.position.orientation,
                            line_width=self.options.goal_state_linewidth
                        )
                    else:
                        viewer.draw_circle(
                            creator=self.__class__.__name__,
                            origin=goal_state.position.center,
                            radius=goal_state.position.radius if hasattr(goal_state.position, 'radius') else 0.75,
                            fill_color=goal_state_color,
                            border_color=goal_state_color,
                            line_width=self.options.goal_state_linewidth
                        )
                        draw_rectangle = False

                if self.options.render_start_waypoints:
                    if draw_rectangle and not self.options.circles:
                        viewer.draw_2d_shape(
                            creator=self.__class__.__name__,
                            vertices=vertices,
                            fill_color=initial_state_color,
                            border_color=initial_state_color,
                            translation=planning_problem.initial_state.position,
                            rotation=planning_problem.initial_state.orientation,
                            line_width=self.options.initial_state_linewidth
                        )
                    else:
                        viewer.draw_circle(
                            creator=self.__class__.__name__,
                            origin=planning_problem.initial_state.position,
                            radius=planning_problem.initial_state.position.radius if hasattr(planning_problem.initial_state.position, 'radius') else 0.75,
                            fill_color=initial_state_color,
                            line_width=self.options.initial_state_linewidth
                        )

            if self.options.render_trajectory:
                planned_ego_trajectory = params.ego_vehicle.ego_route.planning_problem_path
                if planned_ego_trajectory is not None:
                    trajectory_polyline = params.ego_vehicle.ego_route.planning_problem_path_polyline
                    position = params.ego_vehicle.state.position
                    arclength_to_position = trajectory_polyline.get_projected_arclength(position,
                                                                                        linear_projection=False)
                    cut_polylines = cut_polyline(
                        line=planned_ego_trajectory,
                        distance=arclength_to_position
                    )

                    viewer.draw_polyline(
                        creator=self.__class__.__name__,
                        vertices=cut_polylines[0],
                        is_closed=False,
                        color=self.options.trajectory_completed_color,
                        line_width=self.options.trajectory_linewidth
                    )

                    if len(cut_polylines) > 1:
                        viewer.draw_polyline(
                            creator=self.__class__.__name__,
                            vertices=cut_polylines[1],
                            is_closed=False,
                            color=self.options.trajectory_uncompleted_color,
                            line_width=self.options.trajectory_linewidth,
                        )
