import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams, T_Frame
from commonroad_geometric.rendering.video_recording import save_images_from_frames, save_video_from_frames
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace
from commonroad_geometric.simulation.ego_simulation.control_space.implementations import SteeringAccelerationSpace
from commonroad_geometric.simulation.ego_simulation.control_space.keyboard_input import UserQuitInterrupt, UserResetInterrupt, get_keyboard_action
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation, EgoVehicleSimulationFinishedException
from commonroad_geometric.simulation.ego_simulation.respawning.base_respawner import BaseRespawner
from commonroad_geometric.simulation.ego_simulation.respawning.implementations.random_respawner import RandomRespawner
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions


class EgoVehicleGame(AutoReprMixin):
    def __init__(
        self,
        scenario: Union[Scenario, str],
        simulation: Optional[BaseSimulation] = None,
        control_space: Optional[BaseControlSpace] = None,
        respawner: Optional[BaseRespawner] = None,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        traffic_extractor_options: Optional[TrafficExtractorOptions] = None,
        traffic_renderer_options: Optional[TrafficSceneRendererOptions] = None,
        **sumo_simulation_kwargs: Any,
    ) -> None:
        control_space = control_space or SteeringAccelerationSpace()
        respawner = respawner or RandomRespawner()
        traffic_extractor_options = traffic_extractor_options or TrafficExtractorOptions(edge_drawer=VoronoiEdgeDrawer())
        if isinstance(scenario, str):
            scenario, planning_problem_set = CommonRoadFileReader(filename=scenario).open()

        self._renderer = TrafficSceneRenderer(
            options=traffic_renderer_options,
        )
        self._running: bool = False

        if simulation is None:
            simulation = SumoSimulation(
                initial_scenario=scenario,
                options=SumoSimulationOptions(
                    **sumo_simulation_kwargs,
                )
            )
        # simulation._options.on_step_renderer = self._renderer

        traffic_extractor = TrafficExtractor(
            simulation=simulation,
            options=traffic_extractor_options
        )

        self._ego_sim = EgoVehicleSimulation(
            simulation=simulation,
            control_space=control_space,
            respawner=respawner,
            traffic_extractor=traffic_extractor,
        )
        self._video_frames: List[T_Frame] = []

    @property
    def ego_simulation(self) -> EgoVehicleSimulation:
        return self._ego_sim

    @property
    def current_state(self) -> State:
        return self._ego_sim.ego_vehicle.state

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ego_vehicle(self) -> EgoVehicle:
        return self._ego_sim.ego_vehicle

    @property
    def ego_collision(self) -> bool:
        return self._ego_sim.check_if_has_collision().collision

    @property
    def ego_reached_goal(self) -> bool:
        return self._ego_sim.check_if_has_reached_goal()

    @property
    def renderer(self) -> TrafficSceneRenderer:
        return self._renderer

    def start(self) -> None:
        self._ego_sim.start()
        self._running = True

    def close(self) -> None:
        self._ego_sim.close()
        self._running = False

    def step(self) -> None:
        try:
            action = get_keyboard_action(renderer=self.renderer)
        except UserResetInterrupt:
            self._ego_sim.reset()
            action = np.array([0.0, 0.0], dtype=np.float32)
        except UserQuitInterrupt:
            self.close()
            return
        try:
            self._ego_sim.step(action)
        except EgoVehicleSimulationFinishedException:
            self.close()
        render_kwargs: Dict[str, Any] = dict(
            ego_vehicle_vertices=self.ego_vehicle.vertices,
            overlays={
                "Scenario": self._ego_sim.current_scenario.scenario_id,
                "Timestep": self._ego_sim.current_time_step,
                'Action': action
            },
        )
        self._ego_sim.extract_data()
        frame = self._ego_sim.render(
            renderer=self.renderer,
            return_rgb_array=True,
            render_params=RenderParams(
                render_kwargs=render_kwargs
            )
        )
        self._video_frames.append(frame)

    def save_video(self, output_file: str, save_pngs: bool = False) -> None:
        print(f"Saving video of last {len(self._video_frames)} frames to '{output_file}'")
        save_video_from_frames(frames=self._video_frames, output_file=output_file)
        if save_pngs:
            png_output_dir = os.path.join(os.path.dirname(output_file), 'pngs')
            os.makedirs(png_output_dir, exist_ok=True)
            save_images_from_frames(self._video_frames, output_dir=png_output_dir)
        self.clear_frames()

    def clear_frames(self) -> None:
        self._video_frames = []
