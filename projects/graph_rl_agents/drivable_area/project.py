from pathlib import Path

from commonroad.common.solution import VehicleType
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn
from torch.optim import Adam

from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TemporalTrafficExtractorOptions, TrafficExtractorFactory, TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions, TemporalTrafficExtractor
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.dataset.transformation.implementations.feature_normalization.feature_normalization_transformation import FeatureNormalizationTransformation, \
    FeatureUnnormalizationTransformation
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.project.base_rl_project import BaseRLProject
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.rendering.plugins.cameras import *
from commonroad_geometric.rendering.plugins.implementations import RenderEgoVehiclePlugin, RenderPlanningProblemSetPlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import *
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from projects.geometric_models.drivable_area.models.scenario_drivable_area_model import ScenarioDrivableAreaModel
from projects.geometric_models.drivable_area.project import DrivableAreaFeatureComputers, create_edge_drawer, create_lanelet_graph_conversion_steps, create_scenario_filterers
from projects.graph_rl_agents.drivable_area.utils.encoding_observer import EncodingObserver
from projects.graph_rl_agents.drivable_area.utils.post_processors import EncodingPostProcessor

# Control settings
EGO_VEHICLE_SIMULATION_OPTIONS = EgoVehicleSimulationOptions(
    vehicle_model=VehicleModel.KS,
    vehicle_type=VehicleType.BMW_320i
)

def create_rewarders():
    rewarders = [
        # AccelerationPenaltyRewardComputer(
        #     weight=0.0,
        #     loss_type=RewardLossMetric.L2
        # ),
        CollisionPenaltyRewardComputer(
            penalty=-1.5,
        ),
        # FrictionViolationPenaltyRewardComputer(penalty=-0.01),
        TrajectoryProgressionRewardComputer(
            weight=0.1,
            delta_threshold=0.08
        ),
        ConstantRewardComputer(reward=-0.001),
        #
        ReachedGoalRewardComputer(reward=3.5),
        OvershotGoalRewardComputer(reward=0.0),
        # SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
        StillStandingPenaltyRewardComputer(penalty=-0.05, velocity_threshold=2.0),
        TimeToCollisionPenaltyRewardComputer(weight=0.1), # requires incoming edges
        OffroadPenaltyRewardComputer(penalty=-3.5),
        VelocityPenaltyRewardComputer(
            reference_velocity=28.0,
            weight=0.002,
            loss_type=RewardLossMetric.L2,
            only_upper=True
        ),

        LateralErrorPenaltyRewardComputer(weight=0.0001, loss_type=RewardLossMetric.L1),
        YawratePenaltyRewardComputer(weight=0.01),
        # HeadingErrorPenaltyRewardComputer(
        #     weight=0.01,
        #     loss_type=RewardLossMetric.L2,
        #     wrong_direction_penalty=-0.01
        # )
    ]

    return rewarders

def create_scenario_preprocessors():
    scenario_preprocessors = [
        # VehicleFilterPreprocessor(),
        # RemoveIslandsPreprocessor()
        # SegmentLaneletsPreprocessor(100.0),
        ComputeVehicleVelocitiesPreprocessor(),
        # (DepopulateScenarioPreprocessor(1), 1),
    ]
    return scenario_preprocessors

def create_termination_criteria():
    termination_criteria = [
        OffroadCriterion(),
        # OffrouteCriterion(),
        # CollisionCriterion(),
        ReachedGoalCriterion(),
        OvershotGoalCriterion(),
        # TrafficJamCriterion(),
        # FrictionViolationCriterion()
    ]
    return termination_criteria


class DrivableAreaRLProject(BaseRLProject):
    def configure_experiment(self, cfg: dict) -> RLExperimentConfig:
        observer = EncodingObserver(observation_type=cfg["observation_type"])

        postprocessors = []
        if cfg["enable_feature_normalization"]:
            feature_normalizer_transformation = FeatureNormalizationTransformation(
                params_file_path=cfg["feature_normalization_params_path"]
            )
            postprocessors.append(feature_normalizer_transformation.to_post_processor())

        # from commonroad_geometric.dataset.postprocessing.implementations.rasterized_drivable_area_post_processor import BinaryRasterizedDrivableAreaPostProcessor
        # postprocessors.append(BinaryRasterizedDrivableAreaPostProcessor(
        #     pixel_size=64, 
        #     view_range=70, 
        #     flatten=True, 
        #     include_road_coverage=False, 
        #     only_render_ego=True
        # ))

        if cfg["observation_type"] == "encoding":
            encoder_post_processor = EncodingPostProcessor(
                model_filepath=cfg["encoding_model_path"],
                reload_freq=10000,
                enable_decoding=cfg["render_decoding"]
            )
            postprocessors.append(encoder_post_processor)

        if cfg["enable_feature_normalization"]:
            feature_unnormalizer_transformation = FeatureUnnormalizationTransformation(
                params_file_path=cfg["feature_normalization_params_path"]
            )
            postprocessors.append(feature_unnormalizer_transformation.to_post_processor())

        lanelet_graph_conversion_steps = create_lanelet_graph_conversion_steps(
            enable_waypoint_resampling=cfg["enable_waypoint_resampling"],
            waypoint_density=cfg["lanelet_waypoint_density"]
        )

        renderer_plugins = ScenarioDrivableAreaModel.configure_renderer_plugins()
        renderer_plugins.insert(-2, RenderEgoVehiclePlugin())
        renderer_plugins.insert(-2, RenderPlanningProblemSetPlugin())

        traffic_extractor_factory = TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=create_edge_drawer(cfg["edge_range"]),
                feature_computers=TrafficFeatureComputerOptions(
                    v=DrivableAreaFeatureComputers.v(),
                    v2v=DrivableAreaFeatureComputers.v2v(),
                    l=DrivableAreaFeatureComputers.l(),
                    l2l=DrivableAreaFeatureComputers.l2l(),
                    v2l=DrivableAreaFeatureComputers.v2l(),
                    l2v=DrivableAreaFeatureComputers.l2v()
                ),
                only_ego_inc_edges=cfg["only_ego_inc_edges"],
                assign_multiple_lanelets=True,
                ego_map_radius=cfg["ego_map_radius"],
                include_lanelet_vertices=True
            )
        )

        experiment_config = RLExperimentConfig(
            simulation_cls=ScenarioSimulation,
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_CENTER,
                lanelet_graph_conversion_steps=lanelet_graph_conversion_steps,
                linear_lanelet_projection=True,
                sort_lanelet_assignments=False
            ),
            control_space_cls=PIDControlSpace,
            control_space_options=PIDControlOptions(
                lower_bound_steering=-0.2,
                upper_bound_steering=0.2,
                k_P_orientation=2.0,
                k_yaw_rate=0.5,
                k_P_velocity=3.0
            ),
            # control_space_cls=SteeringAccelerationSpace,
            # control_space_options=SteeringAccelerationControlOptions(),
            

            respawner_cls=RandomRespawner,
            respawner_options=RandomRespawnerOptions(
                random_init_arclength=True,
                random_goal_arclength=True,
                random_start_timestep=True,
                only_intersections=False,
                route_length=(10, 15),
                init_speed=cfg["spawning"]["init_speed"],
                min_goal_distance=None,
                min_goal_distance_l2=cfg["spawning"]["min_goal_distance"],
                max_goal_distance_l2=cfg["spawning"]["max_goal_distance"],
                max_goal_distance=None,
                min_remaining_distance=None,
                max_attempts_outer=50,
                min_vehicle_distance=cfg["spawning"]["min_vehicle_distance"],
                min_vehicle_speed=None,
                min_vehicles_route=None,
                max_attempts_inner=5
            ),
            traffic_extraction_factory=TemporalTrafficExtractorFactory(
                options=TemporalTrafficExtractorOptions(
                    collect_num_time_steps=10,
                    collect_skip_time_steps=0,
                    return_incomplete_temporal_graph=True,
                    add_temporal_vehicle_edges=True,
                    max_time_steps_temporal_edge=10,
                    postprocessors=postprocessors
                ),
                traffic_extractor_factory=traffic_extractor_factory
            ),
            ego_vehicle_simulation_options=EGO_VEHICLE_SIMULATION_OPTIONS,
            rewarder=SumRewardAggregator(create_rewarders()),
            termination_criteria=create_termination_criteria(),
            env_options=RLEnvironmentOptions(
                async_resets=False,
                num_respawns_per_scenario=0,
                loop_scenarios=True,
                preprocessor=chain_preprocessors(*(create_scenario_filterers() + create_scenario_preprocessors())),
                render_on_step=cfg["render_on_step"],
                render_debug_overlays=cfg["render_debug_overlays"],
                renderer_options=TrafficSceneRendererOptions(
                    plugins=renderer_plugins,
                    camera=EgoVehicleCamera(),
                ),
                raise_exceptions=cfg["raise_exceptions"],
                observer=observer
            )
        )
        return experiment_config

    def configure_model(self, cfg: dict, experiment: RLExperiment) -> RLModelConfig:
        enable_representations: bool = cfg["enable_representations"]

        feature_extractor_cls = FlattenExtractor
        feature_extractor_kwargs = {}
        return RLModelConfig(
            agent_cls=PPO,
            agent_kwargs=dict(
                gae_lambda=cfg["gae_lambda"],
                gamma=cfg["gamma"],
                n_epochs=cfg["n_epochs"],
                ent_coef=cfg["ent_coef"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                vf_coef=cfg["vf_coef"],
                max_grad_norm=cfg["max_grad_norm"],
                learning_rate=cfg["learning_rate"],
                clip_range=cfg["clip_range"],
                clip_range_vf=None,
                policy='MultiInputPolicy',
                policy_kwargs=dict(
                    ortho_init=False,
                    log_std_init=-1,
                    net_arch={'vf': [256, 128, 64], 'pi': [256, 128, 64]},
                    activation_fn=nn.Tanh,
                    features_extractor_class=feature_extractor_cls,
                    features_extractor_kwargs=feature_extractor_kwargs,
                    optimizer_class=Adam,
                    optimizer_kwargs=dict(
                        eps=1e-5
                    )
                ),
            ),
        )
