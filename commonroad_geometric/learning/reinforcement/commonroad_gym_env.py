import logging
import os
import timeit
from dataclasses import dataclass
from typing import Dict, Generic, List, Literal, Optional, Sequence, Set, Tuple, TypedDict, Union

import gym
import gym.spaces
import numpy as np

from commonroad_geometric.common.logging import DeferredLogMessage
from commonroad_geometric.common.utils.seeding import get_random_seed, set_global_seed
from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorsInput
from commonroad_geometric.learning.reinforcement._episode_resetter import _EpisodeResetter
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver
from commonroad_geometric.learning.reinforcement.observer.flattened_graph_observer import FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.base_reward_aggregator import BaseRewardAggregator
from commonroad_geometric.learning.reinforcement.termination_criteria import TERMINATION_CRITERIA_SCENARIO_FINISHED
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.rendering.defaults import DEFAULT_FPS
from commonroad_geometric.rendering.traffic_scene_renderer import T_Frame, TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams, SkipRenderInterrupt
from commonroad_geometric.simulation.base_simulation import T_BaseSimulationOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation_factory import EgoVehicleSimulationFactory
from commonroad_geometric.simulation.exceptions import SimulationRuntimeError

logger = logging.getLogger(__name__)


DEFAULT_OBSERVER_CLS = FlattenedGraphObserver


@dataclass
class RLEnvironmentOptions:
    """
    Configuration options for the gym environment.

    Args:
        num_respawns_per_scenario (int): Number of times to reset simulation before continuing with next scenario.
        data_padding_size (int): Padding used when converting Pytorch Geometric data instances to fixed-size tensors. Defaults to 1000.
    """
    always_advance: bool = False
    async_reset_delay: float = 0.05 # TODO: This should be automatically tuned based on mean ep. durations for better performance
    async_resets: bool = True
    auto_update_async_reset_delay: bool = True
    log_step_info: bool = False
    loop_scenarios: bool = True
    num_respawns_per_scenario: int = 0
    observer: Optional[BaseObserver] = None
    raise_exceptions: bool = False
    render_debug_overlays: bool = False # TODO: move render stuff to somewhere else
    render_on_step: bool = False # TODO: Add "render_condition", lambda method for enabling & disabling
    renderer_options: Union[None, TrafficSceneRendererOptions, Sequence[TrafficSceneRendererOptions]] = None
    scenario_prefilters: Optional[Sequence[BaseScenarioFilterer]] = None
    scenario_preprocessors: Optional[T_ScenarioPreprocessorsInput] = None


@dataclass
class RLEnvironmentParams:
    scenario_dir: str
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


class CommonRoadGymEnv(gym.Env, Generic[T_BaseSimulationOptions]):
    """
    Gym environment for training RL agents.
    See https://github.com/openai/gym for reference.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': DEFAULT_FPS
    }

    def __init__(self, params: RLEnvironmentParams):
        """Initializes gym environment"""
        self._scenario_iterator = ScenarioIterator(
            directory=params.scenario_dir,
            loop=params.options.loop_scenarios,
            preprocessors=params.options.scenario_preprocessors,
            prefilters=params.options.scenario_prefilters
        )
        self._ego_vehicle_simulation_factory = params.ego_vehicle_simulation_factory
        self._rewarder = params.rewarder
        self._termination_criteria = params.termination_criteria
        self._options = params.options

        self._termination_reasons: Set[str] = set.union(*(criterion.reasons for criterion in self._termination_criteria))
        self._observation_space: Optional[gym.spaces.Dict] = None
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

        self.seed()

        while True:
            try:
                self._resetter.on_init()
                self._observation_space = self._observer.setup(
                    dummy_data=self.ego_vehicle_simulation.extract_data()
                )
                break
            except Exception as e:
                logger.error(e, exc_info=True)

    @property
    def action_space(self) -> gym.Space:
        if self._ego_vehicle_simulation is None:
            raise AttributeError("self._ego_vehicle_simulation is None")
        return self._ego_vehicle_simulation.control_space.gym_action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
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
    def scenario_iterator(self) -> ScenarioIterator:
        return self._scenario_iterator

    @property
    def current_scenario_id(self) -> str:
        return str(self.ego_vehicle_simulation.current_scenario.scenario_id)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], float, bool, CommonRoadGymStepInfo]:
        start_time = timeit.default_timer()

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
                    data=data
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
                lambda _: f"{self.current_scenario_id} step: t={self._step_counter}, a={numpy_prettify(action, 2)}, r={self._rewarder.avg_reward_step:+.3f}, dt={elapsed:.3f}, nxtrdy={self._resetter.ready}")
            )

        self._step_counter += 1
        self._total_step_counter += 1

        if obs is None:
            logger.warn("obs is None, this should never happen. using last obs as fallback") 
            obs = self._last_obs
        else:
            self._last_obs = obs
        
        assert obs is not None

        if error:
            self._force_advance_next = True

        return obs, self._rewarder.avg_reward_step, done, info # TODO: Which reward to return for multi-substep actions?

    def reset(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        self._last_obs = None
        if self._episode_counter > 0:
            advance_scenario = all((
                self._options.num_respawns_per_scenario >= 0,
                self.ego_vehicle_simulation.reset_count >= self._options.num_respawns_per_scenario,
                len(self._scenario_iterator) > 1
            )) or self._force_advance_next or self._options.always_advance
            self._force_advance_next = False
            logger.debug(f"Resetting gym environment with advance_scenario={advance_scenario}")
            if advance_scenario:
                self._resetter.on_reset()
            else:
                self.ego_vehicle_simulation.reset()
            self._rewarder.reset()
            self._step_counter = 0
        try:
            obs, data = self.observe()
        except Exception as e:
            if self._options.raise_exceptions:
                raise e
            logger.error(e, exc_info=True)
            self._force_advance_next = True
            return self.reset() # TODO
        self._episode_counter += 1
        return obs

    def respawn(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        self._last_obs = None
        self.ego_vehicle_simulation.reset()
        self._rewarder.reset()
        self._step_counter = 0
        obs, data = self.observe()
        return obs

    def render(
        self,
        mode: Literal['human', 'rgb_array'] = 'human',
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
                reward_fraction = reward_component/self._rewarder.highest_abs if self._rewarder.highest_abs > 0 else np.nan
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
            for renderer in self.renderers:
                frame = self.ego_vehicle_simulation.render(
                    renderers=renderer,
                    render_params=RenderParams(
                        render_kwargs=dict(
                            overlays=overlays
                        )
                    ),
                    return_rgb_array=mode == 'rgb_array'
                )
                if return_frame is None:
                    return_frame = frame
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

    def observe(self) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], CommonRoadData]:
        data = self.ego_vehicle_simulation.extract_data()

        if data.v.is_ego_mask.sum().item() != 1:
            if self._last_obs is not None:
                logger.warn(f"Observe call encountered data instance with {data.v.is_ego_mask.sum().item()} ego vehicles. Using last observation as fallback (TODO)") # TODO
                return self._last_obs, data
            else:
                raise SimulationRuntimeError(f"Observe call encountered data instance with {data.v.is_ego_mask.sum().item()} ego vehicles")

        obs = self._observer.observe(
            data=data,
            ego_vehicle_simulation=self.ego_vehicle_simulation
        )
        return obs, data

    def seed(self, seed: Optional[int] = None) -> List[int]:
        seed = seed or get_random_seed()
        set_global_seed(seed)
        self._scenario_iterator.shuffle(seed=seed)
        logger.debug(f"Set environment seed: {seed}")
        return [seed]

    def __del__(self) -> None:
        self.close()
