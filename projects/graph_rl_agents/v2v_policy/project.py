from commonroad.common.solution import VehicleType
from stable_baselines3 import PPO
from torch import nn
from torch.optim import Adam

from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.extraction.traffic import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations.voronoi import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficFeatureComputerOptions
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.observer.flattened_graph_observer import FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.project.base_rl_project import BaseRLProject
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.rendering.plugins import *
from commonroad_geometric.rendering.plugins.render_planning_problem_set_plugin import RenderPlanningProblemSetPlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.steering_acceleration_control_space import SteeringAccelerationControlOptions, SteeringAccelerationSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import RandomRespawner, RandomRespawnerOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from projects.graph_rl_agents.v2v_policy.feature_extractor import VehicleGraphFeatureExtractor

RENDERER_OPTIONS = [
    TrafficSceneRendererOptions(
        plugins=[
            RenderLaneletNetworkPlugin(lanelet_linewidth=0.64),
            RenderPlanningProblemSetPlugin(
                render_trajectory=False,
                render_start_waypoints=False,
                render_goal_waypoints=True,
                render_look_ahead_point=False
            ),
            RenderTrafficGraphPlugin(),
            RenderEgoVehiclePlugin(
                render_trail=False,
            ),
            RenderObstaclesPlugin(
                from_graph=False,
                randomize_color_from_lanelet=False
            )
        ],
        camera_follow=True,
        view_range=200.0,
        camera_auto_rotation=True
    ),
]

SCENARIO_PREPROCESSORS = [
    #VehicleFilterPreprocessor(),
    #RemoveIslandsPreprocessor()
    #SegmentLaneletsPreprocessor(25.0)
    #(DepopulateScenarioPreprocessor(1), 1),
]
SCENARIO_PREFILTERS = [
    #TrafficFilterer(),
    #LaneletNetworkSizeFilterer(10)
]
POSTPROCESSORS = [
    #VirtualEgoLaneletPostProcessor(lanelet_length=50.0),
    #RemoveEgoLaneletConnectionsPostProcessor(),
    # LaneletOccupancyPostProcessor(
    #     time_horizon=60,
    # ),
]

# Reinforcement learning problem configuration
REWARDER_COMPONENTS = [
    AccelerationPenaltyRewardComputer(weight=0.0, loss_type=RewardLossMetric.L2),
    CollisionPenaltyRewardComputer(
        penalty=-2.0,
        # not_at_fault_penalty=-0.75,
        speed_multiplier=False,
        max_speed=15.0,
        speed_bias=3.0,
    ),
    #FrictionViolationPenaltyRewardComputer(penalty=-0.01),
    TrajectoryProgressionRewardComputer(
        weight=0.06,
        delta_threshold=0.08
    ),
    #ConstantRewardComputer(reward=-0.001),
    ReachedGoalRewardComputer(reward=2.0),
    #SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
    StillStandingPenaltyRewardComputer(penalty=-0.001, velocity_threshold=2.0),
    #TimeToCollisionPenaltyRewardComputer(weight=0.1), # requires incoming edges
    #YawratePenaltyRewardComputer(weight=0.01)
    VelocityPenaltyRewardComputer(
        reference_velocity=17.0,
        weight=0.002,
        loss_type=RewardLossMetric.L1,
        only_upper=True
    ),
    HeadingErrorPenaltyRewardComputer(weight=0.02, loss_type=RewardLossMetric.L2),
    #LaneChangeRewardComputer(weight=0.2, dense=False),
    LateralErrorPenaltyRewardComputer(weight=0.02, loss_type=RewardLossMetric.L1),
    OffroutePenaltyRewardComputer(penalty=-2.0),
]

TERMINATION_CRITERIA = [
    # OffroadCriterion(),
    OffrouteCriterion(),
    CollisionCriterion(),
    ReachedGoalCriterion(),
    TrafficJamCriterion(),
    # FrictionViolationCriterion()
]

# Data extraction
V_FEATURE_COMPUTERS = [
    ft_veh_state,
    GoalAlignmentComputer(
        include_goal_distance_longitudinal=True,
        include_goal_distance_lateral=False,
        include_goal_distance=True,
        include_lane_changes_required=True,
        logarithmic=False
    ),
    YawRateFeatureComputer(),
    VehicleLaneletPoseFeatureComputer(
        include_longitudinal_abs=True,
        include_longitudinal_rel=True,
        include_lateral_left=False,
        include_lateral_right=False,
        include_lateral_error=True,
        include_heading_error=True,
        update_exact_interval=1
    ),
    VehicleLaneletConnectivityComputer(),
    EgoFramePoseFeatureComputer(),
    NumLaneletAssignmentsFeatureComputer()
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
            simulation_cls=ScenarioSimulation,
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_CENTER,
            ),
            control_space_cls=SteeringAccelerationSpace,
            control_space_options=SteeringAccelerationControlOptions(),
            respawner_cls=RandomRespawner,
            respawner_options=RandomRespawnerOptions(
                random_init_arclength=True,
                random_goal_arclength=True,
                random_start_timestep=True,
                only_intersections=False,
                route_length=(1, 4),
                init_speed=4.0,
                min_goal_distance=None,
                max_goal_distance=None,
                min_remaining_distance=None,
                max_attempts_outer=50,
                min_vehicle_distance=None,
                min_vehicle_speed=None,
                min_vehicles_route=None,
                max_attempts_inner=5
            ),
            traffic_extraction_options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=cfg["dist_threshold_v2v"]),
                feature_computers=TrafficFeatureComputerOptions(
                    v=V_FEATURE_COMPUTERS,
                    v2v=V2V_FEATURE_COMPUTERS,
                    l=L_FEATURE_COMPUTERS,
                    l2l=L2L_FEATURE_COMPUTERS,
                    v2l=V2L_FEATURE_COMPUTERS,
                ),
                postprocessors=POSTPROCESSORS,
                only_ego_inc_edges=False, # set to True to speed up extraction for 1-layer GNNs
                assign_multiple_lanelets=True,
                ego_map_radius=cfg["ego_map_radius"]
            ),
            ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
                vehicle_model=VehicleModel.KS,
                vehicle_type=VehicleType.BMW_320i
            ),
            rewarder=SumRewardAggregator(REWARDER_COMPONENTS),
            termination_criteria=TERMINATION_CRITERIA,
            env_options=RLEnvironmentOptions(
                async_resets=True,
                num_respawns_per_scenario=1,
                loop_scenarios=True,
                scenario_preprocessors=SCENARIO_PREPROCESSORS,
                scenario_prefilters=SCENARIO_PREFILTERS,
                render_on_step=cfg["render_on_step"],
                render_debug_overlays=cfg["render_debug_overlays"],
                renderer_options=RENDERER_OPTIONS,
                raise_exceptions=True,
                observer = FlattenedGraphObserver(data_padding_size=cfg["data_padding_size"])
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
                    net_arch=[{'vf': [256, 128, 64], 'pi': [256, 128, 64]}],
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
                        normalization=False
                    ),
                    optimizer_class=Adam,
                    optimizer_kwargs=dict(
                        eps=1e-5
                    )
                ),
            ),            
        )
