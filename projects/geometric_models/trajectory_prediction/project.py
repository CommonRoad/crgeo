import functools
from pathlib import Path
from typing import Type

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.common.io_extensions.scenario_files import filter_max_scenarios
from commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import LaneletGraphConverter
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.generation.scenario_traffic_generation import generate_traffic
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations import LaneletGraphFilter, WeaklyConnectedFilter
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import SegmentLaneletsPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.log_exception_wrapper import LogExceptionWrapper
from commonroad_geometric.dataset.transformation.implementations.feature_normalization.feature_normalization_transformation import FeatureNormalizationTransformation
from commonroad_geometric.learning.base_project import register_run_command
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.geometric.project.base_geometric_project import BaseGeometricProject
from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputerContainerService, CallbackComputersContainer
from commonroad_geometric.learning.geometric.training.callbacks.implementations.export_latest_model_callback import ExportLatestModelCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.gradient_clipping_callback import GradientClippingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.log_wandb_callback import LogWandbCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.watch_model_callback import WatchWandbCallback
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment, GeometricExperimentConfig
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.implementations.constant_rate_spawner import ConstantRateSpawner
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations.heuristic_scenario_filter import HeuristicOSMScenarioFilter


class FeatureComputers:

    @staticmethod
    def v():
        return [
            ft_veh_state,
            VehicleLaneletConnectivityComputer(),
        ]

    @staticmethod
    def v2v():
        return [
            ft_rel_state_ego,
            LaneletDistanceFeatureComputer(
                max_lanelet_distance=60.0,
                max_lanelet_distance_placeholder=60.0 * 1.1,
            ),
        ]

    @staticmethod
    def l():
        return [

        ]

    @staticmethod
    def l2l():
        return [

        ]

    @staticmethod
    def v2l():
        return [
            VehicleLaneletPoseEdgeFeatureComputer(
                include_longitudinal_abs=True,  # distance to lanelet start
                include_longitudinal_rel=True,  # percent of lanelet distance covered
                include_lateral_left=True,  # lateral distance to left lanelet boundary
                include_lateral_right=True,  # lateral distance to right lanelet boundary
                include_lateral_error=True,  # signed lateral distance to lanelet center (positive = vehicle is to the right of lanelet center)
                include_heading_error=True,  # orientation difference between lanelet and vehicle
                update_exact_interval=1,
            ),
            # TODO relative position & orientation to lanelet start?
        ]

    @staticmethod
    def l2v():
        return [

        ]


def create_scenario_filterers():
    return LogExceptionWrapper(
        wrapped_preprocessor=chain_preprocessors(*[
            WeaklyConnectedFilter(),
            LaneletGraphFilter(min_edges=8, min_nodes=10),
            HeuristicOSMScenarioFilter(),
        ])
    )


def create_edge_drawer(edge_range: float):
    return FullyConnectedEdgeDrawer(dist_threshold=edge_range)


def create_lanelet_graph_conversion_steps(
    enable_waypoint_resampling: bool,
    waypoint_density: int
):
    conversion_steps = [
        LaneletGraphConverter.connect_predecessor_successors,
        LaneletGraphConverter.connect_successor_predecessor,
        LaneletGraphConverter.compute_lanelet_width,
        functools.partial(
            LaneletGraphConverter.lanelet_curvature,
            alpha=0.6,
            max_depth=2,
            lanelet_curvature_aggregation="abs",
            curvature_aggregation="avg",
        ),
        LaneletGraphConverter.lanelet_centered_coordinate_system,
    ]
    if enable_waypoint_resampling:
        conversion_steps.insert(
            0,
            functools.partial(
                LaneletGraphConverter.resample_waypoints,
                waypoint_density=waypoint_density
            )
        )
    return conversion_steps


class TrajectoryPredictionProject(BaseGeometricProject):

    @register_run_command
    def generate_dataset(self) -> None:
        cfg = self.cfg.experiment

        traffic_spawner = ConstantRateSpawner(
            p_spawn=cfg["dataset_generation"]["sumo_simulation"]["p_spawn"]
        )
        sumo_simulation_options = SumoSimulationOptions(
            dt=cfg["dataset_generation"]["sumo_simulation"]["delta_time"],
            collision_checking=False,
            presimulation_steps=cfg["dataset_generation"]["sumo_simulation"]["presimulation_steps"],
            p_wants_lane_change=cfg["dataset_generation"]["sumo_simulation"]["p_wants_lane_change"],
            traffic_spawner=traffic_spawner
        )
        scenario_iterator = ScenarioIterator(
            directory=cfg["dataset_generation"]["input_directory"],
            filter_scenario_paths=functools.partial(filter_max_scenarios,
                                                    max_scenarios=cfg["dataset_generation"]["max_scenarios"])
        )
        generate_traffic(
            input_scenario_iterator=scenario_iterator,
            sumo_simulation_options=sumo_simulation_options,
            scenario_output_dir=cfg["dataset_generation"]["output_directory"],
            time_steps_per_run=cfg["dataset_generation"]["time_steps_per_run"],
            num_workers=cfg["dataset_generation"]["num_workers"],
            time_step_cutoff=30000,
            complete_trajectory_count=0,
            include_ego_vehicle_trajectory=False,
            initial_recorded_time_step=0,
            min_trajectory_length=0,
            overwrite=cfg["dataset_generation"]["overwrite"],
            should_mute_workers=True
        )

    # TODO: Implement as transformation
    # @register_run_command
    # def add_sample_weights(self) -> None:
    #     for dataset in [
    #         self.get_train_dataset(collect_missing_samples=False, throw_missing=False), 
    #         self.get_test_dataset(collect_missing_samples=False, throw_missing=False)
    #     ]:
    #         if dataset is not None:
    #             add_sample_weights(
    #                 dataset=dataset,
    #                 lanelet_sampling_weights_num_bins=self.cfg.experiment["pre_transform"]["lanelet_sampling_weights_num_bins"]
    #             )        

    def configure_experiment(self, cfg: Config) -> GeometricExperimentConfig:
        temporal_enabled: bool = cfg.temporal.enabled

        traffic_extractor_factory = TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=create_edge_drawer(cfg["edge_range"]),
                # Update cfg.model.graph_features when changing feature computers
                feature_computers=TrafficFeatureComputerOptions(
                    v=FeatureComputers.v(),
                    v2v=FeatureComputers.v2v(),
                    l=FeatureComputers.l(),
                    l2l=FeatureComputers.l2l(),
                    v2l=FeatureComputers.v2l(),
                    l2v=FeatureComputers.l2v()
                ),
                assign_multiple_lanelets=True,
                postprocessors=[]
            ),
        )

        extractor_factory: BaseExtractorFactory
        if temporal_enabled:
            extractor_factory = TemporalTrafficExtractorFactory(
                options=TemporalTrafficExtractorOptions(
                    collect_num_time_steps=cfg.temporal.collect_time_steps,
                    collect_skip_time_steps=cfg.temporal.collect_skip_time_steps,
                    return_incomplete_temporal_graph=False,
                    add_temporal_vehicle_edges=cfg["add_temporal_vehicle_edges"],
                    # temporal_vehicle_edge_feature_computers=FeatureComputers.vtv()
                    # max_time_steps_temporal_edge=Unlimited if cfg.temporal.max_time_steps_temporal_edge == "unlimited" else cfg.temporal.max_time_steps_temporal_edge
                ),
                traffic_extractor_factory=traffic_extractor_factory,
            )

        else:
            extractor_factory = traffic_extractor_factory

        lanelet_graph_conversion_steps = create_lanelet_graph_conversion_steps(
            enable_waypoint_resampling=cfg["enable_waypoint_resampling"],
            waypoint_density=cfg["lanelet_waypoint_density"]
        )

        if cfg["enable_feature_normalization"]:
            transformations = [
                FeatureNormalizationTransformation(
                    max_fit_samples=cfg["feature_normalization_max_fit_samples"],
                    params_file_path=cfg["feature_normalization_params_path"],
                    ignore_keys={
                        ("vehicle", "pos"),
                        ("vehicle", "orientation"),
                        ("lanelet", "pos"),
                        ("lanelet", "orientation")
                    }
                )
            ]
        else:
            transformations = []

        experiment_config = GeometricExperimentConfig(
            extractor_factory=extractor_factory,
            dataset_collector_cls=DatasetCollector,
            scenario_preprocessor=create_scenario_filterers() >> SegmentLaneletsPreprocessor(
                    lanelet_max_segment_length=cfg["pre_transform"]["lanelet_max_segment_length"]
                ),
            transformations=transformations,
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_SHAPE,
                collision_checking=False,
                lanelet_graph_conversion_steps=lanelet_graph_conversion_steps,
                step_renderers=[TrafficSceneRenderer()] if cfg["render_collection"] else None
            )
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
                # LogDrivableAreaWandb(wandb_service=wandb_service),
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
        from projects.geometric_models.trajectory_prediction.models.temporal_scenario_models import MultiplePredictionWrapper, TemporalTrajectoryPredictionModel
        from functools import partial
        return partial(MultiplePredictionWrapper, module=TemporalTrajectoryPredictionModel(cfg=cfg))
