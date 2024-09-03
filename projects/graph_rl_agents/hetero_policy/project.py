import random

from commonroad.common.solution import VehicleType
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
from torch import nn
from torch.optim import Adam
import time
from functools import partial

from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.postprocessing.implementations import *
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import (TrafficExtractorOptions,
                                                                               TrafficFeatureComputerOptions)

from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.observer.implementations.ego_enhanced_graph_observer import EgoEnhancedGraphObserver
from commonroad_geometric.learning.reinforcement.project.base_rl_project import BaseRLProject
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations.reached_goal_criterion import OvershotGoalCriterion
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.implementations import *
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_extrapolation import RenderObstacleExtrapolation
from commonroad_geometric.simulation.ego_simulation.control_space.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import (RandomRespawner,
                                                                                       RandomRespawnerOptions)
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (ScenarioSimulation,
                                                                                   ScenarioSimulationOptions)
from commonroad_geometric.simulation.interfaces.static.unpopulated_simulation import UnpopulatedSimulation
from projects.geometric_models.drivable_area.project import create_lanelet_graph_conversion_steps
from projects.graph_rl_agents.hetero_policy.model.feature_extractor import HeteroFeatureExtractor
from projects.graph_rl_agents.common.callbacks.traffic_density_curriculum_callback import TrafficDensityCurriculumCallback

RENDERER_OPTIONS = [
    TrafficSceneRendererOptions(
        plugins=[
            RenderLaneletNetworkPlugin(
                from_graph=True, # dynamic ego map
                lanelet_linewidth=0.3,
                persistent=False # dynamic ego map
            ),
            RenderPlanningProblemSetPlugin(
                render_trajectory=True,
                render_start_waypoints=True,
                render_goal_waypoints=True,
                render_look_ahead_point=True
            ),
            RenderTrafficGraphPlugin(),
            RenderEgoVehiclePlugin(
                direction_arrow=True,
                trail_arrows=True
            ),
            #RenderEgoVehicleInputPlugin(),
            # RenderCompassRosePlugin(),
            RenderObstaclePlugin(
                from_graph=True,
                randomize_color_from=None
            ),
            #RenderObstacleExtrapolation(),
            # RenderDrivableAreaPlugin(alpha=0.25),
            # RenderTrafficFlowLaneletNetworkPlugin()
            # RenderVehicleToLaneletEdgesPlugin()
            # RenderLaneletGraphPlugin()
        ],
        camera=EgoVehicleCamera(view_range=200.0, camera_rotation_speed=0.0),
        fps=60
    ),
]

class DepopulateIncreaser:
    def __init__(self, use_time_based: bool = False, start_rate: float = 0.0, increase_rate: float = None) -> None:
        self.use_time_based = use_time_based
        self.start_rate = start_rate
        # If the increase rate is not specified, set it to reach 1.0 after 24 hours
        if increase_rate is None:
            self.increase_rate = 1.0 / (200 * 60 * 60)  # Default increase rate for x hours
        else:
            self.increase_rate = increase_rate
        self.call_count = 0
        self.start_time = time.time()

    def __call__(self) -> float:
        if self.use_time_based:
            elapsed_time = time.time() - self.start_time  # Calculate elapsed time in seconds
            depopulate_value = min(1.0, self.start_rate + elapsed_time * self.increase_rate)
        else:
            depopulate_value = min(1.0, self.start_rate + self.call_count * self.increase_rate)
            self.call_count += 1
        return depopulate_value

POSTPROCESSORS = [
    # VirtualEgoLaneletPostProcessor(lanelet_length=50.0),
    # RemoveEgoLaneletConnectionsPostProcessor(),
    # LaneletOccupancyPostProcessor(
    #     time_horizon=60,
    # ),
]


class HeteroPolicyProject(BaseRLProject):
    def configure_experiment(self, cfg: dict) -> RLExperimentConfig:

        # control_space_cls = SteeringAccelerationSpace
        # control_space_options = SteeringAccelerationControlOptions()
        control_space_cls = PIDControlSpace
        control_space_options = PIDControlOptions()
        # control_space_cls = PIDLaneChangeControlSpace
        # control_space_options = PIDLaneChangeControlOptions()

        termination_criteria = [
            ReachedGoalCriterion(),
            OvershotGoalCriterion(),
        ]

        if cfg["termination"]["collision"]:
            termination_criteria.append(
                CollisionCriterion()
            )
        if cfg["termination"]["offroad"]:
            termination_criteria.append(
                OffroadCriterion()
            )
        if cfg["termination"]["timeout"] is not None:
            termination_criteria.append(
                TimeoutCriterion(max_timesteps=cfg["termination"]["timeout"])
            )
        if cfg["termination"]["traffic_jam"]:
            termination_criteria.append(
                TrafficJamCriterion()
            )

        postprocessors = [RemoveEgoLaneletConnectionsPostProcessor()]
        if not cfg["disable_drivable_area_rasterizing"]:
            from commonroad_geometric.dataset.postprocessing.implementations.rasterized_drivable_area_post_processor import BinaryRasterizedDrivableAreaPostProcessor
            postprocessors.append(BinaryRasterizedDrivableAreaPostProcessor(
                pixel_size=64, 
                view_range=70, 
                flatten=True, 
                include_road_coverage=False, 
                only_render_ego=True
            ))

        if cfg["highway_mode"]:
            postprocessors.append(
                VirtualEgoHighwayPostProcessor(lanelet_length=cfg["virtual_lanelet_length"])
            )

        experiment_config = RLExperimentConfig(
            simulation_cls=ScenarioSimulation if cfg["enable_traffic"] else UnpopulatedSimulation,
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_SHAPE,
                lanelet_graph_conversion_steps=create_lanelet_graph_conversion_steps(
                    enable_waypoint_resampling=cfg["enable_waypoint_resampling"],
                    waypoint_density=cfg["lanelet_waypoint_density"]
                ),
                linear_lanelet_projection=cfg["linear_lanelet_projection"]
            ),
            control_space_cls=control_space_cls,
            control_space_options=control_space_options,
            respawner_cls=RandomRespawner,
            respawner_options=RandomRespawnerOptions(
                random_init_arclength=True,
                random_goal_arclength=True,
                random_start_timestep=True,
                only_intersections=False,
                route_length=(10, 15),
                init_speed="auto",
                min_goal_distance=cfg["spawning"]["min_goal_distance"],
                max_goal_distance=None,
                min_remaining_distance=None,
                max_attempts_outer=50,
                min_vehicle_distance=cfg["spawning"]["min_vehicle_distance"],
                min_vehicle_speed=None,
                min_vehicles_route=None,
                max_attempts_inner=5
            ),
            traffic_extraction_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=TrafficFlowEdgeDrawer(
                        base_edge_drawer=VoronoiEdgeDrawer(),
                        dist_threshold=cfg["dist_threshold_v2v"]
                    ),
                    postprocessors=postprocessors,
                    only_ego_inc_edges=False,  # set to True to speed up extraction for 1-layer GNNs
                    assign_multiple_lanelets=True,
                    ego_map_radius=cfg["ego_map_radius"],
                    ego_map_strict=True,
                    linear_lanelet_projection=cfg["linear_lanelet_projection"],
                    feature_computers=TrafficFeatureComputerOptions(
                        v=[
                            ft_veh_state,
                            YawRateFeatureComputer(),
                            VehicleLaneletConnectivityComputer(),
                            VehicleLaneletPoseFeatureComputer(linear_lanelet_projection=True)
                        ],
                        v2v=[
                            ft_same_lanelet,
                            LaneletDistanceFeatureComputer(
                                linear_lanelet_projection=cfg["linear_lanelet_projection"],
                                max_lanelet_distance_placeholder=60.0 * 1.1,
                                max_lanelet_distance=60.0
                            ),
                            ft_rel_state_ego,
                            TimeToCollisionFeatureComputer(),
                        ],
                        l=[
                            LaneletGeometryFeatureComputer(),
                        ],
                        l2l=[
                            LaneletConnectionGeometryFeatureComputer(),
                        ],
                        v2l=[
                            VehicleLaneletPoseEdgeFeatureComputer(
                                linear_lanelet_projection=cfg["linear_lanelet_projection"]
                            )
                        ]
                    )
                )
            ),
            ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
                vehicle_model=VehicleModel.KS,
                vehicle_type=VehicleType.BMW_320i
            ),
            rewarder=SumRewardAggregator([
                ConstantRewardComputer(reward=cfg["reward"]["living_penalty"]),
                PathFollowingRewardComputer(
                    max_speed=cfg["reward"]["cruise_speed"],
                    cross_track_error_sensitivity=cfg["reward"]["cross_track_error_sensitivity"],
                    coefficient=cfg["reward"]["path_following"],
                ),
                ReachedGoalRewardComputer(
                    reward=cfg["reward"]["reached_goal"]
                ),
                VelocityPenaltyRewardComputer(
                    only_upper=True,
                    reference_velocity=cfg["reward"]["max_speed"],
                    weight=cfg["reward"]["overspeeding_penalty_coefficient"],
                    loss_type=RewardLossMetric.L1
                ),
                CollisionPenaltyRewardComputer(
                    penalty=cfg["reward"]["collision"]
                ),
                TimeToCollisionPenaltyRewardComputer(
                    weight=cfg["reward"]["time_to_collision"]
                ),
                OffroadPenaltyRewardComputer(
                    penalty=cfg["reward"]["offroad"]
                ),
                StillStandingPenaltyRewardComputer(
                    penalty=cfg["reward"]["still_standing"]
                )
            ]),
            termination_criteria=termination_criteria,
            env_options=RLEnvironmentOptions(
                async_resets=cfg["async_resets"],
                num_respawns_per_scenario=cfg["num_respawns_per_scenario"],
                loop_scenarios=True,
                preprocessor=chain_preprocessors(
                    # TrafficFilterer(),
                    # LaneletNetworkSizeFilterer(10)
                    # VehicleFilterPreprocessor(),
                    # RemoveIslandsPreprocessor()
                    # ComputeVehicleVelocitiesPreprocessor(),
                    # SegmentLaneletsPreprocessor(100.0),
                    DepopulateScenarioPreprocessor(depopulator=partial(
                        DepopulateIncreaser, 
                        use_time_based=True,
                        start_rate=cfg.curriculum.vehicle_spawning_start_rate, 
                        # increase_rate=cfg.curriculum.vehicle_spawning_increase_rate
                    ))
                ),
                render_on_step=cfg["render_on_step"],
                render_debug_overlays=cfg["render_debug_overlays"],
                renderer_options=RENDERER_OPTIONS,
                raise_exceptions=cfg["raise_exceptions"],
                observer=EgoEnhancedGraphObserver(
                    data_padding_size=cfg["data_padding_size"],
                    include_graph_observations=cfg["enable_traffic"]
                )
            )
        )
        return experiment_config
    
    def configure_custom_callbacks(self, cfg: dict) -> list[BaseCallback]:
        custom_callbacks = []
        # if cfg.curriculum.enabled:
        #     custom_callbacks.append(
        #         TrafficDensityCurriculumCallback(increase_rate=cfg.curriculum.vehicle_spawning_increase_rate)
        #     )
        return custom_callbacks

    def configure_model(self, cfg: dict, experiment: RLExperiment) -> RLModelConfig:
        return RLModelConfig(
            agent_cls=PPO,
            agent_kwargs=dict(
                gae_lambda=cfg["gae_lambda"],
                gamma=cfg["gamma"],
                n_epochs=cfg["n_epochs"],
                policy_coef=cfg["policy_coef"],
                ent_coef=cfg["ent_coef"],
                vf_coef=cfg["vf_coef"],
                recon_coef=cfg["recon_coef"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                max_grad_norm=cfg["max_grad_norm"],
                learning_rate=cfg["learning_rate"],
                clip_range=cfg["clip_range"],
                clip_range_vf=None,
                policy='MultiInputPolicy',
                policy_kwargs=dict(
                    ortho_init=False,
                    log_std_init=cfg["log_std_init"],
                    net_arch={'vf': cfg["net_arch_vf"], 'pi': cfg["net_arch_pi"]},
                    activation_fn=nn.Tanh,
                    features_extractor_class=HeteroFeatureExtractor,
                    features_extractor_kwargs=dict(
                        encoder_config=cfg["encoder_config"],
                        decoder_config=cfg["decoder_config"],
                        gnn_features=cfg["gnn_features"]
                    ),
                    optimizer_class=Adam,
                    optimizer_kwargs=dict(
                        eps=1e-5
                    )
                ),
            ),
        )
