import gymnasium
import numpy as np
from typing import Optional


from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions


class RenderObserver(BaseObserver):
    def __init__(self, renderer: TrafficSceneRenderer):
        self.renderer = renderer
        super().__init__()

    def setup(self, dummy_data: Optional[CommonRoadData] = None) -> gymnasium.Space:
        return gymnasium.spaces.Box(low=0, high=255, shape=(
            self.renderer.options.viewer_options.window_width, 
            self.renderer.options.viewer_options.window_height, 3), dtype=np.uint8) 

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        # Return the rendered image from the ego vehicle simulation
        obs = ego_vehicle_simulation.render(renderers=[self.renderer], return_frames=True)[0]
        return obs