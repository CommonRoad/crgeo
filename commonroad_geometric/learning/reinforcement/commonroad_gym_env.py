import logging
import os
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Set, Tuple, TypedDict, Union, Iterator

import gymnasium
import numpy as np
from pathlib import Path
from commonroad_geometric.common.logging import DeferredLogMessage
from commonroad_geometric.common.utils.seeding import get_random_seed, set_global_seed
from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.learning.reinforcement._episode_resetter import _EpisodeResetter
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.base_reward_aggregator import BaseRewardAggregator
from commonroad_geometric.learning.reinforcement.termination_criteria import TERMINATION_CRITERIA_SCENARIO_FINISHED
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.rendering.traffic_scene_renderer import DEFAULT_FPS, T_Frame, TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams, SkipRenderInterrupt
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.simulation.base_simulation import T_SimulationOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation_factory import EgoVehicleSimulationFactory
from commonroad_geometric.simulation.exceptions import SimulationRuntimeError

import warnings

# Filter out the specific warning by message
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'gymnasium',
)

# Set global NumPy printing options
np.set_printoptions(precision=3, suppress=True)

logger = logging.getLogger(__name__)

DEFAULT_OBSERVER_CLS = FlattenedGraphObserver
T_InfoDict = dict[str, Any]


@dataclass
class RLEnvironmentOptions:
    """
    Configuration options for the gym environment.

    Args:
        num_respawns_per_scenario (int): Number of times to reset simulation before continuing with next scenario.
        data_padding_size (int): Padding used when converting Pytorch Geometric data instances to fixed-size tensors. Defaults to 1000.
    """
    always_advance: bool = False
    async_reset_delay: float = 0.05  # TODO: This should be automatically tuned based on mean ep. durations for better performance
    async_resets: bool = False
    auto_update_async_reset_delay: bool = True
    disable_graph_extraction: bool = False
    log_step_info: bool = False
    loop_scenarios: bool = True
    num_respawns_per_scenario: int = 0
    observer: Optional[BaseObserver] = None
    raise_exceptions: bool = False
    render_debug_overlays: bool = False  # TODO: move render stuff to somewhere else
    render_on_step: bool = False  # TODO: Add "render_condition", lambda method for enabling & disabling
    renderer_options: Union[None, TrafficSceneRendererOptions, Sequence[TrafficSceneRendererOptions]] = None
    preprocessor: Optional[BaseScenarioPreprocessor] = None


@dataclass
class RLEnvironmentParams:
    scenario_dir: Path
    ego_vehicle_simulation_factory: EgoVehicleSimulationFactory
    rewarder: BaseRewardAggregator
    termination_criteria: List[BaseTerminationCriterion]
    options: RLEnvironmentOptions


class CommonRoadGymStepInfo(TypedDict):
    """
    Info struct summarizing the environment at the current time-step.
    """
    termination_reason: Optional[str]
    reward_components: Dict[str, float]
    scenario_id: str
    current_num_obstacles: int
    sim_time_step: int
    time_step: int
    training_step: int
    total_num_obstacles: int
    reward_component_episode_info: Optional[Dict[str, Dict[str, float]]]
    vehicle_aggregate_stats: Dict[str, Dict[str, float]]
    cumulative_reward: float
    done: bool
    elapsed: float
    next_reset_ready: bool
    highest_reward_computer: str
    highest_reward: float
    lowest_reward_computer: str
    lowest_reward: float


class CommonRoadGymEnv(gymnasium.Env, Generic[T_SimulationOptions]):
    """
    Gym environment for training RL agents.
    See https://github.com/openai/gym for reference.
    """

    metadata = {
        # 'render_modes': ['human', 'rgb_array'],
        # 'video.frames_per_second': DEFAULT_FPS
    }

    def __init__(self, params: RLEnvironmentParams | dict, render_mode: str = 'rgb_array'):
        """Initializes gym environment"""
        self.render_mode = render_mode
        if isinstance(params.options, dict):
            params.options = RLEnvironmentOptions(**params.options)
        self._scenario_iterator = ScenarioIterator(
            directory=Path(params.scenario_dir),
            is_looping=params.options.loop_scenarios,
            preprocessor=params.options.preprocessor,
        )
        self._scenario_iterable = iter(self._scenario_iterator)
        self._ego_vehicle_simulation_factory = params.ego_vehicle_simulation_factory
        self._rewarder = params.rewarder
        self._termination_criteria = params.termination_criteria
        self._options = params.options

        self._termination_reasons: Set[str] = set.union(
            *(criterion.reasons for criterion in self._termination_criteria))
        self._observation_space: Optional[gymnasium.spaces.Dict] = None
        self._ego_vehicle_simulation: Optional[EgoVehicleSimulation] = None

        self._current_action: Optional[np.ndarray] = None
        self._step_counter: int = 0
        self._total_step_counter: int = 0
        self._episode_counter: int = 0
        self._last_obs: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None
        self._force_advance_next: bool = False

        self._renderers: Optional[List[TrafficSceneRenderer]] = None
        self._queued_overlays: Optional[Dict[str, str]] = None
        self._resetter = _EpisodeResetter(
            self,
            async_resets=self._options.async_resets,
            injected_async_delay=self._options.async_reset_delay,
            auto_update_delay=self._options.auto_update_async_reset_delay
        )
        self._observer = self._options.observer if self._options.observer is not None else DEFAULT_OBSERVER_CLS()
        # With removal of seed method from Env API, the seed is now passed to reset
        # Need to keep track of last seed to prevent reshuffling the ScenarioIterator on every reset
        self._last_seed = None

        while True:
            try:
                self._resetter.on_init()
                self._observation_space = self._observer.setup(
                    dummy_data=self.ego_vehicle_simulation.extract_data() if not self.options.disable_graph_extraction else None
                )
                break
            except Exception as e:
                logger.error(e, exc_info=True)

    @property
    def action_space(self) -> gymnasium.Space:
        if self._ego_vehicle_simulation is None:
            raise AttributeError("self._ego_vehicle_simulation is None")
        return self._ego_vehicle_simulation.control_space.gym_action_space

    @property
    def observation_space(self) -> gymnasium.Space:
        if self._observation_space is None:
            raise AttributeError("self._observation_space is None")
        return self._observation_space

    @property
    def options(self) -> RLEnvironmentOptions:
        return self._options

    @property
    def ego_vehicle_simulation(self) -> EgoVehicleSimulation:
        if self._ego_vehicle_simulation is None:
            raise AttributeError("self._ego_vehicle_simulation is None")
        return self._ego_vehicle_simulation

    @property
    def ego_vehicle_simulation_factory(self) -> EgoVehicleSimulationFactory:
        if self._ego_vehicle_simulation_factory is None:
            raise AttributeError("self._ego_vehicle_simulation_factory is None")
        return self._ego_vehicle_simulation_factory

    @property
    def step_counter(self) -> int:
        return self._step_counter

    @property
    def total_step_counter(self) -> int:
        return self._total_step_counter

    @property
    def has_started(self) -> bool:
        return self._total_step_counter > 0

    @property
    def termination_reasons(self) -> Set[str]:
        return self._termination_reasons

    @property
    def scenario_iterable(self) -> Iterator[ScenarioBundle]:
        return self._scenario_iterable

    @property
    def scenario_iterator(self) -> ScenarioIterator:
        return self._scenario_iterator

    @property
    def current_scenario_id(self) -> str:
        return str(self.ego_vehicle_simulation.current_scenario.scenario_id)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[T_Observation, float, bool, bool, CommonRoadGymStepInfo]:
        start_time = timeit.default_timer()
        truncated: bool = False
        done: bool = False
        termination_reason = None

        self._resetter.on_step()
        self._rewarder.reset_step()

        obs = None
        error = False

        # Applying action
        try:
            for _ in self.ego_vehicle_simulation.step(action=action):
                # Retrieving observation dict
                obs, data = self.observe()

                # Calculating reward
                self._rewarder.on_substep(
                    action=action,
                    simulation=self.ego_vehicle_simulation,
                    data=data,
                    observation=obs
                )

                # Evaluating termination criteria
                if not done:
                    for criterion in self._termination_criteria:
                        criterion_active, criterion_reason = criterion(self.ego_vehicle_simulation)
                        done = done or criterion_active
                        if done:
                            termination_reason = criterion_reason
                            break

                if self._options.render_on_step:
                    self.render()

                if done:
                    break
        except Exception as e:
            if self._options.raise_exceptions:
                raise e
            logger.error(e, exc_info=True)
            done = True
            termination_reason = repr(e)
            error = True

        if self.ego_vehicle_simulation.simulation.lifecycle.is_finished:
            done = True
            termination_reason = TERMINATION_CRITERIA_SCENARIO_FINISHED

        if obs is None:
            try:
                obs, data = self.observe()
            except Exception as e:
                if self._options.raise_exceptions:
                    raise e
                logger.error(e, exc_info=True)
                done = True
                termination_reason = repr(e)
                error = True

        elapsed = timeit.default_timer() - start_time
        if termination_reason is not None:
            self._termination_reasons.add(termination_reason)

        highest_reward_computer, highest_reward = self._rewarder.highest
        lowest_reward_computer, lowest_reward = self._rewarder.lowest
        info = CommonRoadGymStepInfo(
            termination_reason=termination_reason,
            reward_components=self._rewarder.reward_component_info_step,
            scenario_id=self.current_scenario_id,
            current_num_obstacles=self.ego_vehicle_simulation.current_num_non_ego_obstacles,
            sim_time_step=self.ego_vehicle_simulation.current_time_step,
            time_step=self.step_counter,
            training_step=self.total_step_counter,
            total_num_obstacles=len(self.ego_vehicle_simulation.current_scenario.dynamic_obstacles) - 1,
            reward_component_episode_info=self._rewarder.component_aggregate_info() if done else None,
            vehicle_aggregate_stats=self.ego_vehicle_simulation.ego_vehicle.aggregate_state_statistics,
            cumulative_reward=self._rewarder.cumulative_reward,
            done=done,
            elapsed=elapsed,
            next_reset_ready=self._resetter.ready,
            highest_reward_computer=highest_reward_computer,
            highest_reward=highest_reward,
            lowest_reward_computer=lowest_reward_computer,
            lowest_reward=lowest_reward
        )

        self._current_action = action

        if done:
            logger.info(
                f"""Episode {self._episode_counter} done(
    reason={termination_reason},
    tot_reward={self._rewarder.cumulative_reward:.3f},
    ep_length={self.step_counter},
    scenario={self.ego_vehicle_simulation.current_scenario.scenario_id},
    tot_steps={self.total_step_counter},
    process={os.getpid()}
)"""
            )
        elif self._options.log_step_info:
            logger.debug(DeferredLogMessage(
                lambda
                    _: f"{self.current_scenario_id} step: t={self._step_counter}, a={numpy_prettify(action, 2)}, r={self._rewarder.avg_reward_step:+.3f}, dt={elapsed:.3f}, nxtrdy={self._resetter.ready}")
            )

        self._step_counter += 1
        self._total_step_counter += 1

        if obs is None:
            logger.warn("obs is None, this should never happen. using last obs as fallback")
            obs = self._last_obs
        else:
            self._last_obs = obs

        assert obs is not None

        if isinstance(obs, dict):
            for key, value in obs.items():
                if np.isnan(value).any():
                    # logger.warn(f"NaN values found in array under key '{key}'. These will be replaced with zero.")
                    value[np.isnan(value)] = 0.0
        else:
            obs[np.isnan(obs)] = 0.0

        if error:
            self._force_advance_next = True
        terminated = done
        # TODO: Which reward to return for multi-substep actions?
        return obs, self._rewarder.avg_reward_step, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None  # Unused, mainly here to adhere to the Env interface
    ) -> tuple[T_Observation, T_InfoDict]:
        if self._last_seed is None:
            self._last_seed = seed or get_random_seed()
            set_global_seed(self._last_seed)
            # self._scenario_iterator.shuffle(seed=self._last_seed)
            # self._scenario_iterable = iter(self._scenario_iterator)
            logger.debug(f"Set environment seed: {self._last_seed}")
        self._last_obs = None
        if self._episode_counter > 0:
            advance_scenario = all((
                self._options.num_respawns_per_scenario >= 0,
                self.ego_vehicle_simulation.reset_count >= self._options.num_respawns_per_scenario,
                self._scenario_iterator.max_result_scenarios > 1
            )) or self._force_advance_next or self._options.always_advance
            self._force_advance_next = False
            logger.debug(f"Resetting gym environment with advance_scenario={advance_scenario}")
            if advance_scenario:
                self._resetter.on_reset()
            else:
                self.ego_vehicle_simulation.reset()
            self._rewarder.reset()
            self._observer.reset(self.ego_vehicle_simulation)
            self._step_counter = 0
        try:
            obs, data = self.observe()
        except Exception as e:
            if self._options.raise_exceptions:
                raise e
            logger.error(e, exc_info=True)
            self._force_advance_next = True
            return self.reset()  # TODO
        self._episode_counter += 1
        info = {}
        return obs, info

    def respawn(self) -> T_Observation:
        self._last_obs = None
        self.ego_vehicle_simulation.reset()
        self._rewarder.reset()
        self._step_counter = 0
        obs, data = self.observe()
        return obs

    def render(
        self,
        mode: Literal['human', 'rgb_array'] = 'rgb_array',
        **render_kwargs: Any
    ) -> T_Frame:
        extra_overlays = self._queued_overlays
        self._queued_overlays = None

        if self._options.render_debug_overlays:
            highest_reward_computer, highest_reward = self._rewarder.highest
            lowest_reward_computer, lowest_reward = self._rewarder.lowest
            overlays = {
                'Scenario': f"{str(self.ego_vehicle_simulation.current_scenario.scenario_id)}",
                'Time-step': f"{self.ego_vehicle_simulation.current_time_step}/{self.ego_vehicle_simulation.final_time_step} (ep. {self._episode_counter})",
                'Action': self._current_action,
                'Velocity': self.ego_vehicle_simulation.ego_vehicle.state.velocity,
                'Substep reward': self._rewarder.reward_substep,
                'Min. Step reward': self._rewarder.min_reward_step,
                'Max. Step reward': self._rewarder.max_reward_step,
                'Avg. Step reward': self._rewarder.avg_reward_step,
                'Cum. Step reward': self._rewarder.cumulative_reward_step,
                'Lowest reward': f"{lowest_reward_computer} ({lowest_reward:.3f})",
                'Highest reward': f"{highest_reward_computer} ({highest_reward:.3f})",
                'Episode reward': self._rewarder._cumulative_reward,
                'Ego Lanelet': self.ego_vehicle_simulation.current_lanelet_ids
            }
            for reward_name, reward_component in self._rewarder.reward_component_info_step.items():
                reward_fraction = reward_component / self._rewarder.highest_abs if self._rewarder.highest_abs > 0 else np.nan
                overlays[reward_name] = f"{reward_component:.3f} ({reward_fraction:.1%})"
            # cached_data = self.ego_vehicle_simulation.extract_data(use_cached=True)
            # for key in cached_data.ego.feature_columns:
            #     value = cached_data.ego[key]
            #     overlays[key.title()] = value.numpy() if isinstance(value, Tensor) else value
        else:
            overlays = {}

        if extra_overlays is not None:
            overlays.update(extra_overlays)

        return_frame: T_Frame = None
        try:
            for renderer in self.unwrapped.renderers:
                frame = self.ego_vehicle_simulation.render(
                    renderers=[renderer],
                    render_params=RenderParams(
                        render_kwargs=dict(
                            overlays=overlays,
                            **render_kwargs
                        )
                    ),
                    return_frames=mode == 'rgb_array',
                )
                if return_frame is None:
                    return_frame = frame[0]
        except SkipRenderInterrupt:
            logger.info(f".render received SkipRenderInterrupt, skipping rendering.")

        return return_frame

    @property
    def renderers(self) -> List[TrafficSceneRenderer]:
        if self._renderers is not None:
            return self._renderers

        renderer_options: Sequence[Optional[TrafficSceneRendererOptions]]

        if self._options.renderer_options is None:
            renderer_options = []
        elif isinstance(self._options.renderer_options, (dict, TrafficSceneRendererOptions)):
            renderer_options = [self._options.renderer_options]
        else:
            renderer_options = self._options.renderer_options

        self._renderers = [TrafficSceneRenderer(options=o) for o in renderer_options]

        return self._renderers

    def close(self) -> None:
        if self._ego_vehicle_simulation is not None:
            self._ego_vehicle_simulation.close()

    def observe(self) -> Tuple[T_Observation, CommonRoadData]:
        if self.options.disable_graph_extraction:
            data = None
        else:
            data = self.ego_vehicle_simulation.extract_data()

        obs = self._observer.observe(
            data=data,
            ego_vehicle_simulation=self.ego_vehicle_simulation
        )
        self._verify_observation_dimensions(obs)
        return obs, data
    
    def _verify_observation_dimensions(self, obs):
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            # Get the expected shape from observation_space
            expected_shape = self.observation_space.shape
            # Check if the shape of the observation matches the expected shape
            if obs.shape != expected_shape:
                raise ValueError(f"Dimension mismatch for {key}: expected {expected_shape}, got {value.shape}")
        elif isinstance(self.observation_space, gymnasium.spaces.Dict):
            observation_space = dict(self.observation_space)
            # Iterate through the keys and values in the obs dictionary
            for key, value in obs.items():
                # Check if the key from obs is in the observation_space
                if key in observation_space:
                    # Get the expected shape from observation_space
                    expected_shape = self.observation_space[key].shape
                    # Check if the shape of the observation matches the expected shape
                    if value.shape != expected_shape:
                        raise ValueError(f"Dimension mismatch for {key}: expected {expected_shape}, got {value.shape}")
                else:
                    # If the key is not found in observation_space, raise an error
                    raise KeyError(f"{key} is not a valid observation type.")


    def seed(self, seed: Optional[int] = None) -> List[int]:
        seed = seed or get_random_seed()
        set_global_seed(seed)
        logger.debug(f"Set environment seed: {seed}")
        return [seed]

    def __del__(self) -> None:
        self.close()
