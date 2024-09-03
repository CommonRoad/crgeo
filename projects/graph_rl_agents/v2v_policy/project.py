from commonroad.common.solution import VehicleType
from stable_baselines3 import PPO
from torch import nn
from torch.optim import Adam

from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.project.base_rl_project import BaseRLProject
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from projects.geometric_models.drivable_area.project import create_lanelet_graph_conversion_steps
from commonroad_geometric.learning.reinforcement.observer.implementations.ego_enhanced_graph_observer import EgoEnhancedGraphObserver
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.rendering.plugins.cameras.follow_vehicle_camera import FollowVehicleCamera
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.implementations import *
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.steering_acceleration_control_space import SteeringAccelerationControlOptions, SteeringAccelerationSpace
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.pid_control_space import PIDControlOptions, PIDControlSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import (RandomRespawner,
                                                                                       RandomRespawnerOptions)
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (ScenarioSimulation,
                                                                                   ScenarioSimulationOptions)
from commonroad_geometric.simulation.interfaces.static.unpopulated_simulation import UnpopulatedSimulation
from projects.graph_rl_agents.v2v_policy.feature_extractor import VehicleGraphFeatureExtractor

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
        ReachedGoalRewardComputer(reward=5.0),
        OvershotGoalRewardComputer(reward=-0.5),
        # SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
        StillStandingPenaltyRewardComputer(penalty=-0.05, velocity_threshold=2.0),
        # LateralErrorPenaltyRewardComputer(weight=0.0001, loss_type=RewardLossMetric.L1),
        TimeToCollisionPenaltyRewardComputer(weight=0.1), # requires incoming edges
        # YawratePenaltyRewardComputer(weight=0.01),
        OffroadPenaltyRewardComputer(penalty=-3.0),
        VelocityPenaltyRewardComputer(
            reference_velocity=18.0,
            weight=0.002,
            loss_type=RewardLossMetric.L2,
            only_upper=True
        ),
        # HeadingErrorPenaltyRewardComputer(
        #     weight=0.01,
        #     loss_type=RewardLossMetric.L2,
        #     wrong_direction_penalty=-0.01
        # )
    ]

    return rewarders

def create_scenario_filterers():
    return [
        # WeaklyConnectedFilter(),
        # LaneletGraphFilter(
        #     min_edges=8,
        #     min_nodes=10
        # ),
        # HeuristicOSMScenarioFilter(),
    ]

def create_scenario_preprocessors():
    scenario_preprocessors = [
        # VehicleFilterPreprocessor(),
        # RemoveIslandsPreprocessor()
        SegmentLaneletsPreprocessor(100.0),
        ComputeVehicleVelocitiesPreprocessor()
        # (DepopulateScenarioPreprocessor(1), 1),
    ]
    return scenario_preprocessors

def create_termination_criteria():
    termination_criteria = [
        OffroadCriterion(),
        # OffrouteCriterion(),
        CollisionCriterion(),
        ReachedGoalCriterion(),
        OvershotGoalCriterion(),
        TrafficJamCriterion(),
        # FrictionViolationCriterion()
    ]
    return termination_criteria



RENDERER_OPTIONS = [
    TrafficSceneRendererOptions(
        camera=EgoVehicleCamera(view_range=200.0),
        plugins=[
            RenderLaneletNetworkPlugin(lanelet_linewidth=0.64),
            RenderPlanningProblemSetPlugin(
                render_trajectory=True,
                render_start_waypoints=True,
                render_goal_waypoints=True,
                render_look_ahead_point=True
            ),
            RenderTrafficGraphPlugin(),
            RenderEgoVehiclePlugin(
                render_trail=False,
            ),
            RenderObstaclePlugin(
                from_graph=False,
            ),
            RenderOverlayPlugin()
        ],
    ),
]

# Data extraction
V_FEATURE_COMPUTERS = [
    ft_veh_state,
    GoalAlignmentComputer(
        include_goal_distance_longitudinal=True,
        include_goal_distance_lateral=True,
        include_goal_distance=True,
        include_lane_changes_required=True,
        logarithmic=False,
        closeness_transform=True
    ),
    YawRateFeatureComputer(),
    VehicleLaneletPoseFeatureComputer(
        include_longitudinal_abs=False,
        include_longitudinal_rel=False,
        include_lateral_left=True,
        include_lateral_right=True,
        include_lateral_error=False,
        include_heading_error=True,
        update_exact_interval=10
    ),
    # # VehicleLaneletConnectivityComputer(),
    EgoFramePoseFeatureComputer(),
    # NumLaneletAssignmentsFeatureComputer(),
    DistanceToRoadBoundariesFeatureComputer()
]
L_FEATURE_COMPUTERS = [
    LaneletGeometryFeatureComputer(),
]
L2L_FEATURE_COMPUTERS = [
    LaneletConnectionGeometryFeatureComputer(),
]
V2V_FEATURE_COMPUTERS = [
    ClosenessFeatureComputer(),
    TimeToCollisionFeatureComputer(),
    ft_rel_state_ego,
]
V2L_FEATURE_COMPUTERS = [
    VehicleLaneletPoseEdgeFeatureComputer(update_exact_interval=1)
]


class V2VPolicyProject(BaseRLProject):
    def configure_experiment(self, cfg: dict) -> RLExperimentConfig:
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
            control_space_cls=PIDControlSpace,
            control_space_options=PIDControlOptions(),
            respawner_cls=RandomRespawner,
            respawner_options=RandomRespawnerOptions(
                random_init_arclength=True,
                random_goal_arclength=True,
                random_start_timestep=True,
                only_intersections=False,
                route_length=(10, 15),
                init_speed=10.0,
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
            traffic_extraction_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=VoronoiEdgeDrawer(
                        dist_threshold=cfg["dist_threshold_v2v"]
                    ) if cfg["edge_drawer_class_name"] != "KNearestEdgeDrawer" else KNearestEdgeDrawer(
                        k=cfg["edge_drawer_k"],
                        dist_threshold=cfg["dist_threshold_v2v"],
                    ),
                    feature_computers=TrafficFeatureComputerOptions(
                        v=V_FEATURE_COMPUTERS,
                        v2v=V2V_FEATURE_COMPUTERS,
                        l=L_FEATURE_COMPUTERS,
                        l2l=L2L_FEATURE_COMPUTERS,
                        v2l=V2L_FEATURE_COMPUTERS,
                    ),
                    postprocessors=[],
                    only_ego_inc_edges=False,  # set to True to speed up extraction for 1-layer GNNs
                    assign_multiple_lanelets=True,
                    ego_map_radius=cfg["ego_map_radius"]
                )
            ),
            ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
                vehicle_model=VehicleModel.KS,
                vehicle_type=VehicleType.BMW_320i
            ),
            rewarder=SumRewardAggregator(create_rewarders()),
            termination_criteria=create_termination_criteria(),
            env_options=RLEnvironmentOptions(
                async_resets=False,
                num_respawns_per_scenario=0,
                loop_scenarios=True,
                preprocessor=chain_preprocessors(*(create_scenario_filterers() + create_scenario_preprocessors())),
                render_on_step=cfg["render_on_step"],
                render_debug_overlays=cfg["render_debug_overlays"],
                renderer_options=RENDERER_OPTIONS,
                raise_exceptions=cfg["raise_exceptions"],
                observer=FlattenedGraphObserver(data_padding_size=cfg["data_padding_size"]),
            )
        )
        return experiment_config

    def configure_model(self, cfg: dict, experiment: RLExperiment) -> RLModelConfig:
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
                    features_extractor_class=VehicleGraphFeatureExtractor,
                    features_extractor_kwargs=dict(
                        gnn_hidden_dim=cfg["gnn_hidden_dim"],
                        gnn_layers=cfg["gnn_layers"],
                        gnn_out_dim=cfg["gnn_out_dim"],
                        concat_ego_features=True,
                        self_loops=False,
                        aggr='max',
                        activation_fn=nn.Tanh,
                        normalization=False,
                        weight_decay=0.001
                    ),
                    optimizer_class=Adam,
                    optimizer_kwargs=dict(
                        eps=1e-5
                    )
                ),
            ),
        )
