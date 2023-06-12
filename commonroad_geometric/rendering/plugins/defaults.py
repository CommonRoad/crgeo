from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.plugins.render_ego_vehicle_input_plugin import RenderEgoVehicleInputPlugin
from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.render_obstacles_plugin import RenderObstaclesPlugin
from commonroad_geometric.rendering.plugins.render_ego_vehicle_plugin import RenderEgoVehiclePlugin
from commonroad_geometric.rendering.plugins.render_traffic_graph_plugin import RenderTrafficGraphPlugin
from commonroad_geometric.rendering.plugins.render_planning_problem_set_plugin import RenderPlanningProblemSetPlugin

from typing import List, Optional, Type

DEFAULT_RENDERING_PLUGINS = [
    RenderLaneletNetworkPlugin(), 
    RenderTrafficGraphPlugin(), 
    RenderObstaclesPlugin(), 
    RenderEgoVehiclePlugin(),
    RenderPlanningProblemSetPlugin(), 
    RenderEgoVehicleInputPlugin(),
]

def insert_renderer_plugin_after(insert: BaseRendererPlugin, at: Type[BaseRendererPlugin], plugins: Optional[List[BaseRendererPlugin]] = None):
    plugins = plugins or DEFAULT_RENDERING_PLUGINS
    renderer_plugins = []
    for plugin in plugins:
        renderer_plugins.append(plugin)
        if isinstance(plugin, at):
            renderer_plugins.append(insert)
    return renderer_plugins

def insert_renderer_plugin_before(insert: BaseRendererPlugin, at: Type[BaseRendererPlugin], plugins: Optional[List[BaseRendererPlugin]] = None):
    plugins = plugins or DEFAULT_RENDERING_PLUGINS
    renderer_plugins = []
    for plugin in plugins:
        if isinstance(plugin, at):
            renderer_plugins.append(insert)
        renderer_plugins.append(plugin)
    return renderer_plugins

def remove_renderer_plugin(remove: Type[BaseRendererPlugin], plugins: Optional[List[BaseRendererPlugin]] = None):
    plugins = plugins or DEFAULT_RENDERING_PLUGINS
    renderer_plugins = []
    for plugin in plugins:
        renderer_plugins.append(plugin)
        if not isinstance(plugin, remove):
            renderer_plugins.append(plugin)
    return renderer_plugins
