from typing import List, Optional, Type

from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.plugins.implementations.render_ego_vehicle_input_plugin import RenderEgoVehicleInputPlugin
from commonroad_geometric.rendering.plugins.implementations.render_ego_vehicle_plugin import RenderEgoVehiclePlugin
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.implementations.render_planning_problem_set_plugin import RenderPlanningProblemSetPlugin
from commonroad_geometric.rendering.plugins.implementations.render_traffic_graph_plugin import RenderTrafficGraphPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin

DEFAULT_RENDERING_PLUGINS = [
    RenderLaneletNetworkPlugin(),
    RenderTrafficGraphPlugin(),
    RenderObstaclePlugin(),
    RenderEgoVehiclePlugin(),
    RenderPlanningProblemSetPlugin(),
    RenderEgoVehicleInputPlugin(),
]


def insert_renderer_plugin_after(
    insert: BaseRenderPlugin,
    at: Type[BaseRenderPlugin],
    plugins: Optional[List[BaseRenderPlugin]] = None
):
    plugins = plugins or DEFAULT_RENDERING_PLUGINS
    renderer_plugins = []
    for plugin in plugins:
        renderer_plugins.append(plugin)
        if isinstance(plugin, at):
            renderer_plugins.append(insert)
    return renderer_plugins


def insert_renderer_plugin_before(
    insert: BaseRenderPlugin,
    at: Type[BaseRenderPlugin],
    plugins: Optional[List[BaseRenderPlugin]] = None
):
    plugins = plugins or DEFAULT_RENDERING_PLUGINS
    renderer_plugins = []
    for plugin in plugins:
        if isinstance(plugin, at):
            renderer_plugins.append(insert)
        renderer_plugins.append(plugin)
    return renderer_plugins


def remove_renderer_plugin(
    remove: Type[BaseRenderPlugin],
    plugins: Optional[List[BaseRenderPlugin]] = None
):
    plugins = plugins or DEFAULT_RENDERING_PLUGINS
    renderer_plugins = []
    for plugin in plugins:
        renderer_plugins.append(plugin)
        if not isinstance(plugin, remove):
            renderer_plugins.append(plugin)
    return renderer_plugins
