from commonroad.common.solution import VehicleType
from stable_baselines3 import PPO
from torch import nn
from torch.optim import Adam
from pathlib import Path

from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import (TrafficExtractorOptions,
                                                                               TrafficFeatureComputerOptions)
from commonroad_geometric.dataset.postprocessing.implementations import *
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.observer.implementations import *
from commonroad_geometric.learning.reinforcement.project.base_rl_project import BaseRLProject
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.implementations import (RenderEgoVehiclePlugin,
                                                                    RenderLaneletNetworkPlugin,
                                                                    RenderPlanningProblemSetPlugin)
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import *
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (ScenarioSimulation,
                                                                                   ScenarioSimulationOptions)
from projects.geometric_models.lane_occupancy.models.occupancy.occupancy_model import OccupancyModel
from projects.geometric_models.lane_occupancy.utils.renderer_plugins import RenderLaneletOccupancyPredictionPlugin
from projects.graph_rl_agents.highway_lane_rep.feature_extractor import HighwayLaneRepFeatureExtractor
from projects.graph_rl_agents.lane_occupancy.utils.occupancy_encoding_post_processor import OccupancyEncodingPostProcessor


SCENARIO_PREPROCESSORS = [
    # VehicleFilterPreprocessor(),
    # RemoveIslandsPreprocessor()
    # SegmentLaneletsPreprocessor(25.0)
    # (DepopulateScenarioPreprocessor(1), 1),
]
SCENARIO_PREFILTERS = [
    # TrafficFilterer(),
    # LaneletNetworkSizeFilterer(10)
]

# Control settings
EGO_VEHICLE_SIMULATION_OPTIONS = EgoVehicleSimulationOptions(
    vehicle_model=VehicleModel.KS,
    vehicle_type=VehicleType.BMW_320i
)

TERMINATION_CRITERIA = [
    OffroadCriterion(),
    # OffrouteCriterion(),
    CollisionCriterion(),
    ReachedGoalCriterion(),
    # TrafficJamCriterion(),
    # FrictionViolationCriterion()
]


class HighwayLaneRepRLProject(BaseRLProject):
    def configure_experiment(self, cfg: dict) -> RLExperimentConfig:
        postprocessors = [
            RemoveEgoLaneletConnectionsPostProcessor(),
            VirtualEgoHighwayPostProcessor(lanelet_length=cfg["virtual_lanelet_length"]),
            # TODO missing?
            OccupancyEncodingPostProcessor(
               model_cls=OccupancyModel,
               model_filepath=Path(cfg["model_path"]).resolve()
            )
            # VirtualEgoLaneletPostProcessor(lanelet_length=50.0)
        ]

        experiment_config = RLExperimentConfig(
            simulation_cls=ScenarioSimulation,
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_SHAPE,
                linear_lanelet_projection=cfg["linear_lanelet_projection"]
            ),
            control_space_cls=SteeringAccelerationSpace,
            control_space_options=SteeringAccelerationControlOptions(
            ),
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
            traffic_extraction_options=TrafficExtractorOptions(
                edge_drawer=NoEdgeDrawer(),
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
                        # NumLaneletAssignmentsFeatureComputer(),
                    ],
                    v2v=[
                        ft_same_lanelet,
                        LaneletDistanceFeatureComputer(
                            linear_lanelet_projection=cfg["linear_lanelet_projection"]
                        ),
                        ft_rel_state_ego
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
            ),
            ego_vehicle_simulation_options=EGO_VEHICLE_SIMULATION_OPTIONS,
            rewarder=SumRewardAggregator([
                AccelerationPenaltyRewardComputer(weight=0.0, loss_type=RewardLossMetric.L2),
                CollisionPenaltyRewardComputer(
                    penalty=-2.0,
                    # not_at_fault_penalty=-0.75,
                    speed_multiplier=False,
                    max_speed=15.0,
                    speed_bias=3.0,
                ),
                # FrictionViolationPenaltyRewardComputer(penalty=-0.01),
                TrajectoryProgressionRewardComputer(
                    weight=0.06,
                    delta_threshold=0.08,
                    linear_path_projection=cfg["linear_lanelet_projection"]
                ),
                # ConstantRewardComputer(reward=-0.001),
                #
                ReachedGoalRewardComputer(reward=2.0),
                # SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
                StillStandingPenaltyRewardComputer(penalty=-0.001, velocity_threshold=2.0),

                # TimeToCollisionPenaltyRewardComputer(weight=0.1), # requires incoming edges
                # YawratePenaltyRewardComputer(weight=0.01)
                VelocityPenaltyRewardComputer(
                    reference_velocity=17.0,
                    weight=0.002,
                    loss_type=RewardLossMetric.L1,
                    only_upper=True
                ),
            ]),
            termination_criteria=TERMINATION_CRITERIA,
            env_options=RLEnvironmentOptions(
                async_resets=False,
                num_respawns_per_scenario=0,
                loop_scenarios=True,
                preprocessor=chain_preprocessors(*(SCENARIO_PREFILTERS + SCENARIO_PREPROCESSORS)),
                render_on_step=cfg["render_on_step"],
                render_debug_overlays=cfg["render_debug_overlays"],
                renderer_options=[
                    TrafficSceneRendererOptions(
                        plugins=[
                            RenderLaneletNetworkPlugin(from_graph=True, persistent=False),
                            RenderObstaclePlugin(from_graph=True),
                            RenderEgoVehiclePlugin(),
                            RenderPlanningProblemSetPlugin(
                                render_trajectory=False
                            ),
                            RenderLaneletOccupancyPredictionPlugin(
                                enable_matplotlib_plot=False,
                                recreate_polyline=True
                            )
                        ],
                        camera=EgoVehicleCamera(view_range=200.0),
                    )
                ],
                raise_exceptions=cfg["raise_exceptions"],
                observer=EgoEnhancedGraphObserver(
                    data_padding_size=cfg["data_padding_size"],
                    include_path_observations=False
                )
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
                    features_extractor_class=HighwayLaneRepFeatureExtractor,
                    features_extractor_kwargs={},
                    optimizer_class=Adam,
                    optimizer_kwargs=dict(
                        eps=1e-5
                    )
                ),
            ),
        )
