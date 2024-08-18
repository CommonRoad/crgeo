from pathlib import Path
from typing import Type

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import SegmentLaneletsPreprocessor
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.geometric.project.base_geometric_project import BaseGeometricProject
from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputerContainerService, CallbackComputersContainer
from commonroad_geometric.learning.geometric.training.callbacks.implementations.export_latest_model_callback import ExportLatestModelCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.gradient_clipping_callback import GradientClippingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.log_wandb_callback import LogWandbCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.watch_model_callback import WatchWandbCallback
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment, GeometricExperimentConfig
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from projects.geometric_models.lane_occupancy.models.occupancy.occupancy_model import OccupancyModel


class HighwayLaneRepProject(BaseGeometricProject):

    def configure_experiment(self, cfg: Config) -> GeometricExperimentConfig:
        extractor_factory = TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=NoEdgeDrawer(),
                assign_multiple_lanelets=True,
                postprocessors=[],
                ignore_unassigned_vehicles=True
            ),
        )

        experiment_config = GeometricExperimentConfig(
            extractor_factory=extractor_factory,
            dataset_collector_cls=DatasetCollector,
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_SHAPE,
                collision_checking=False,
                step_renderers=[
                    TrafficSceneRenderer(
                        options=TrafficSceneRendererOptions(
                            plugins=[
                                RenderLaneletNetworkPlugin(from_graph=False),
                                RenderObstaclePlugin(from_graph=False)
                            ]
                        )
                    )
                ] if cfg["render_collection"] else None
            ),
            scenario_preprocessor=SegmentLaneletsPreprocessor(
                lanelet_max_segment_length=cfg["pre_transform"]["lanelet_max_segment_length"]
            ),
            transformations=[],
        )

        return experiment_config

    def configure_training_callbacks(
        self,
        wandb_service: WandbService,
        model_dir: Path
    ) -> CallbackComputersContainer:
        callbacks_computers = CallbackComputersContainer(
            training_step_callbacks=CallbackComputerContainerService([
                ExportLatestModelCallback(
                    directory=model_dir,
                    save_frequency=self.cfg.training.checkpoint_frequency,
                    only_best=False
                ),
                # LogWandbCallback(wandb_service=wandb_service),
                GradientClippingCallback(self.cfg.training.gradient_clipping_threshold)
                # DebugTrainBackwardGradientsCallback(frequency=200)
            ]),
            validation_step_callbacks=CallbackComputerContainerService([

            ]),
            test_step_callbacks=CallbackComputerContainerService([
                LogWandbCallback(wandb_service=wandb_service),
            ]),
            logging_callbacks=CallbackComputerContainerService([LogWandbCallback(wandb_service=wandb_service)]),
            initialize_training_callbacks=CallbackComputerContainerService([WatchWandbCallback(
                wandb_service=wandb_service,
                log_freq=self.cfg.training.log_freq,
                log_gradients=False  # not self.cfg.warmstart
            )]),
            # checkpoint_callbacks=CallbackComputerContainerService([EpochCheckpointCallback(
            #     directory=self.dir_structure.model_dir,
            # )]),
            # early_stopping_callbacks=CallbackComputerContainerService([EarlyStoppingCallback(
            #     after_epochs=self.cfg.training.early_stopping
            # )])
        )
        return callbacks_computers

    def configure_model(self, cfg: Config, experiment: GeometricExperiment) -> Type[BaseGeometric]:
        return OccupancyModel
