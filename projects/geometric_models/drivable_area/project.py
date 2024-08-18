import functools
from pathlib import Path
from typing import Type, List, Optional, Union
import torch
import logging
from copy import deepcopy
import wandb

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad_geometric.common.config import Config
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.common.io_extensions.scenario_files import filter_max_scenarios
from commonroad_geometric.dataset.collection.base_dataset_collector import BaseDatasetCollector
from commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from commonroad_geometric.dataset.collection.temporal_dataset_collector import TemporalDatasetCollector
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import LaneletGraphConverter
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TrafficExtractorFactory
from commonroad_geometric.dataset.postprocessing.implementations import *
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations import LaneletGraphFilter, WeaklyConnectedFilter
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.dataset.transformation.implementations.feature_normalization import FeatureNormalizationTransformation
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
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.implementations.constant_rate_spawner import ConstantRateSpawner
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from projects.geometric_models.drivable_area.functions.visualize_road_coverage import visualize_road_coverage
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations.heuristic_scenario_filter import HeuristicOSMScenarioFilter
from projects.geometric_models.drivable_area.utils.log_drivable_area_callback import LogDrivableAreaWandb, LogOccupancyFlowWandb, LogDrivableAreaTemporalWandb, LogOccupancyFlowTemporalWandb

from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.transformation.dataset_transformation import dataset_transformation

from commonroad_geometric.dataset.transformation.base_transformation import BaseDataTransformation
from commonroad_geometric.dataset.transformation.implementations.feature_normalization.normalize_features import FeatureNormalizer, normalize_features, get_normalization_params_file_path
from projects.geometric_models.drivable_area.utils.vectorized_occupancy_post_processor import save_compute_occupancy_vectorized
from projects.geometric_models.common.multiple_predictions_wrapper import MultiplePredictionWrapper


class ComputeOccupancyTransformation(BaseDataTransformation):
    def __init__(
        self,
        num_workers: int = 1,
        image_size: int = 64
    ) -> None:
        self.num_workers = num_workers
        self.image_size = image_size

    def transform(self, scenario_index: int, sample_index: int, data: CommonRoadData):
        save_compute_occupancy_vectorized(data=data, image_size=self.image_size)  
        yield data

    def _transform_dataset(self, dataset: CommonRoadDataset) -> CommonRoadDataset:
        dataset_transformation(
            dataset=dataset,
            transform=self.transform,
            num_workers=self.num_workers
        )
        return dataset

    def _transform_data(self, data: CommonRoadData) -> CommonRoadData:
        return self.transform(0, 0, data)


class RemoveCloneConnectionsPostprocessor(BaseDataPostprocessor):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:        
        for data in samples:
            node_mask = torch.where(data.v.is_clone == 1)[0]
            edge_mask = (data.v2v.edge_index[0, :, None] == node_mask).any(dim=1)
            data.v2v.edge_index = data.v2v.edge_index[:, ~edge_mask]
            for k, v in data.v2v._mapping.items():
                if k == "edge_index":
                    continue
                data.v2v[k] = data.v2v[k][~edge_mask, :]
            
        return samples

class VehicleCheckIfCloneFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation
    ) -> FeatureDict:
        return {
            "is_clone": params.obstacle.__dict__.get('_is_clone', False)
        }


class DrivableAreaFeatureComputers:

    @staticmethod
    def v():
        return [
            VehicleCheckIfCloneFeatureComputer(),
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
            TimeToCollisionFeatureComputer(),
        ]

    @staticmethod
    def l():
        return []

    @staticmethod
    def l2l():
        return []

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
                update_exact_interval=10,
            ),
            # TODO relative position & orientation to lanelet start?
        ]

    @staticmethod
    def l2v():
        return []


def create_scenario_filterers():
    return [
        # WeaklyConnectedFilter(),
        # LaneletGraphFilter(
        #     min_edges=8,
        #     min_nodes=10
        # ),
        # HeuristicOSMScenarioFilter(),
    ]

def create_scenario_preprocessors(lanelet_max_segment_length: float):
    return [
        ComputeVehicleVelocitiesPreprocessor(),
        # CloneVehicleTrajectoriesPreprocessor(
        #     position_noise=10.0,
        #     orientation_noise="uniform",
        #     velocity_noise=10.0
        # ),
        SegmentLaneletsPreprocessor(lanelet_max_segment_length=lanelet_max_segment_length)
    ]

def create_edge_drawer(edge_range: float):
    return VoronoiEdgeDrawer(dist_threshold=edge_range)


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


class DrivableAreaProject(BaseGeometricProject):

    @register_run_command
    def generate_scenarios(self) -> None:
        cfg = self.cfg.experiment

        from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
        from commonroad_geometric.dataset.scenario.generation.scenario_traffic_generation import generate_traffic
        from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulationOptions

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
            filter_scenario_paths=functools.partial(filter_max_scenarios, max_scenarios=cfg["dataset_generation"]["max_scenarios"]),
            preprocessor=chain_preprocessors(*(create_scenario_filterers() + create_scenario_preprocessors(lanelet_max_segment_length=50.0))),
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
            should_mute_workers=True,
            render_generation=cfg["dataset_generation"]["render"],
        )

    @register_run_command
    def visualize_road_coverage(self) -> None:
        # This needs to be refactored
        dataset = self.get_dataset(
            scenario_dir=self.cfg.dataset.train_scenario_dir,
            dataset_dir=self.cfg.project_dir,
            force_dataset_overwrite=False
        )
        visualize_road_coverage(
            dataset=dataset,
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

        if cfg.disable_drivable_area_rasterizing:
            postprocessors = []
        else:
            # postprocessors = [Lanelet2DOccupancyPostProcessor(
            #     time_horizon=1
            # )]
            postprocessors = [
                RemoveCloneConnectionsPostprocessor(),
                # BinaryRasterizedDrivableAreaPostProcessor(
                #     lanelet_fill_offset=cfg["lanelet_fill_offset"],
                #     lanelet_fill_resolution=cfg["lanelet_fill_resolution"],
                #     pixel_size=cfg["pixel_size"],
                #     view_range=cfg["view_range"],
                #     remove_ego=cfg["remove_ego"],
                #     only_incoming_edges=cfg["only_incoming_edges"],
                #     include_road_coverage=False
                # )
            ]

        traffic_extractor_factory = TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=create_edge_drawer(cfg["edge_range"]),
                # Update cfg.model.graph_features when changing feature computers
                feature_computers=TrafficFeatureComputerOptions(
                    v=DrivableAreaFeatureComputers.v(),
                    v2v=DrivableAreaFeatureComputers.v2v(),
                    l=DrivableAreaFeatureComputers.l(),
                    l2l=DrivableAreaFeatureComputers.l2l(),
                    v2l=DrivableAreaFeatureComputers.v2l(),
                    l2v=DrivableAreaFeatureComputers.l2v()
                ),
                assign_multiple_lanelets=True,
                postprocessors=postprocessors
            ),
        )

        extractor_factory: BaseExtractorFactory
        data_collector_cls: Type[BaseDatasetCollector]
        if temporal_enabled:
            extractor_factory = TemporalTrafficExtractorFactory(
                options=TemporalTrafficExtractorOptions(
                    collect_num_time_steps=cfg.temporal.collect_time_steps,
                    collect_skip_time_steps=cfg.temporal.collect_skip_time_steps,
                    return_incomplete_temporal_graph=False,
                    add_temporal_vehicle_edges=cfg["add_temporal_vehicle_edges"],
                    # max_time_steps_temporal_edge=Unlimited if cfg.temporal.max_time_steps_temporal_edge == "unlimited" else cfg.temporal.max_time_steps_temporal_edge
                ),
                traffic_extractor_factory=traffic_extractor_factory,
            )
            data_collector_cls = TemporalDatasetCollector
        else:
            extractor_factory = traffic_extractor_factory
            data_collector_cls = DatasetCollector

        lanelet_graph_conversion_steps = create_lanelet_graph_conversion_steps(
            enable_waypoint_resampling=cfg["enable_waypoint_resampling"],
            waypoint_density=cfg["lanelet_waypoint_density"]
        )

        if cfg["enable_feature_normalization"]:
            transformations = [
                FeatureNormalizationTransformation(
                    max_fit_samples=1000,
                    params_file_path=cfg["feature_normalization_params_path"]
                )
            ]
        else:
            transformations = []
        
        transformations.append(
            ComputeOccupancyTransformation(
                num_workers=cfg["num_workers"],
                image_size=cfg["pixel_size"]
            )
        )

        experiment_config = GeometricExperimentConfig(
            extractor_factory=extractor_factory,
            dataset_collector_cls=data_collector_cls,
            scenario_preprocessor=chain_preprocessors(
                *(create_scenario_filterers() + create_scenario_preprocessors(lanelet_max_segment_length=cfg["pre_transform"]["lanelet_max_segment_length"]))
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
                LogDrivableAreaTemporalWandb(wandb_service=wandb_service),
                LogOccupancyFlowTemporalWandb(wandb_service=wandb_service),
                LogDrivableAreaTemporalWandb(wandb_service=wandb_service, target_attribute="polar_occupancy"),
                LogOccupancyFlowTemporalWandb(wandb_service=wandb_service, occupancy_attribute="polar_occupancy", target_attribute="polar_occupancy_flow"),
            ]),
            test_step_callbacks=CallbackComputerContainerService([
                LogWandbCallback(wandb_service=wandb_service),
            ]),
            logging_callbacks=CallbackComputerContainerService([
                LogWandbCallback(wandb_service=wandb_service)
            ]),
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
        from projects.geometric_models.drivable_area.models.scenario_drivable_area_model import ScenarioDrivableAreaModel
        from projects.geometric_models.drivable_area.models.scenario_temporal_drivable_area_model import  ScenarioTemporalDrivableAreaModel
        from projects.geometric_models.drivable_area.models.road_network_model import RoadCoveragePredictionModel
        from functools import partial
                                    
        return ScenarioTemporalDrivableAreaModel



    @register_run_command
    def plot(self) -> None:
        from projects.geometric_models.drivable_area.models.scenario_drivable_area_model import plot_all_vehicles
        from commonroad_geometric.common.utils.filesystem import get_most_recent_file, list_files, load_dill, search_file
        model_path = self._get_latest_model_path()
        model = load_dill(model_path)
        model = model.to('cpu')

        dataset = self.get_train_dataset()
        for data in dataset:
            idx, encodings, predictions = model(data)
            plot_all_vehicles(data, predictions)
            continue

    def train_model(self, config=None):
        with wandb.init(config=config):
            # Copy your config so that you don't modify the original
            cfg = deepcopy(self.cfg)

            # Update the config with values from the wandb sweep
            cfg.model['drivable_area_decoder']['lstm_hidden_size'] = wandb.config.lstm_hidden_size
            cfg.model['drivable_area_decoder']['lstm_num_layers'] = wandb.config.lstm_num_layers
            cfg.model['drivable_area_decoder']['temporal_attention_heads'] = wandb.config.temporal_attention_heads
            cfg.model['drivable_area_decoder']['decoder_type'] = wandb.config.decoder_type
            cfg.model['drivable_area_decoder']['conv_channels'] = wandb.config.conv_channels
            cfg.model['drivable_area_decoder']['use_residual_blocks'] = wandb.config.use_residual_blocks
            cfg.model['drivable_area_decoder']['use_dice_loss'] = wandb.config.use_dice_loss
            cfg.model['drivable_area_decoder']['bce_weight'] = wandb.config.bce_weight

            # Adjust training settings
            cfg.training.max_epochs = wandb.config.tuning_epochs
            cfg.training.early_stopping = wandb.config.early_stopping
            cfg.wandb_logging = False

            # Create a new project instance with the updated config
            trial_project = DrivableAreaProject(cfg)

            # Train the model using the existing train method
            context = trial_project.train()

            # Retrieve the best validation loss
            best_val_loss = min(context.losses['validation']['best'])

            # Log the result to wandb
            wandb.log({"best_val_loss": best_val_loss})

    @register_run_command
    def tune_hyperparameters(self) -> None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        sweep_configuration = {
            'method': 'bayes',
            'name': 'drivable-area-sweep',
            'metric': {'goal': 'minimize', 'name': 'best_val_loss'},
            'parameters': {
                'lstm_hidden_size': {'values': [128, 256, 512]},
                'lstm_num_layers': {'min': 1, 'max': 3},
                'temporal_attention_heads': {'values': [4, 8, 16]},
                'decoder_type': {'values': ['ConvTranspose', 'Upsample']},
                'conv_channels': {'values': [[128, 64, 32, 16], [256, 128, 64, 32], [64, 32, 16, 8]]},
                'use_residual_blocks': {'values': [True, False]},
                'use_dice_loss': {'values': [True, False]},
                'bce_weight': {'min': 0.3, 'max': 0.7},
                'tuning_epochs': {'value': self.cfg.additional_config.get('hyperparameter_tuning', {}).get('tuning_epochs', 100)},
                'early_stopping': {'value': self.cfg.additional_config.get('hyperparameter_tuning', {}).get('early_stopping', 100)},
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="drivable-area-tuning")

        logger.info(f"Starting wandb sweep with id: {sweep_id}")
        logger.info(f"Sweep configuration: {sweep_configuration}")

        wandb.agent(sweep_id, function=self.train_model, count=self.cfg.additional_config.get('hyperparameter_tuning', {}).get('n_trials', 100))

        logger.info("Hyperparameter tuning completed.")