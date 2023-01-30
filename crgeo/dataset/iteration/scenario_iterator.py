from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union, cast

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.common.io_extensions.scenario import find_scenario_files
from crgeo.common.logging import stdout
from crgeo.common.utils.filesystem import load_pickle, save_pickle
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from crgeo.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorsInput, T_ScenarioPreprocessorsPipeline

logger = logging.getLogger(__name__)


@dataclass
class ScenarioBundle:
    """
    Container object yielded by ScenarioIterator.
    """
    preprocessed_scenario: Scenario
    input_scenario: Scenario
    input_scenario_file: Path
    load_scenario_from_pickle: bool
    preprocessed_planning_problem_set: Optional[PlanningProblemSet]
    planning_problem_set: Optional[PlanningProblemSet]
    internal_index: int
    trajectory_pickle_file: Optional[Path]  # binary file with postfix .pkl


class ScenarioIterationError(RuntimeError):
    pass


class ScenarioIterator(AutoReprMixin):
    """
    Class for iterating over CommonRoad scenarios found in folder.

    Example usage::

        scenario_iterator = ScenarioIterator(
            directory='scenarios/highD'
        )
        for scenario in scenario_iterator:
            print(scenario)

    """

    def __init__(
        self,
        directory: Union[Path, str],
        loop: bool = False,
        skip_subvariants: bool = False,
        verbose: int = 2,
        shuffle: bool = False,
        save_scenario_pickles: bool = True,
        load_scenario_pickles: bool = True,
        preprocessors: Optional[T_ScenarioPreprocessorsInput] = None,
        prefilters: Optional[Sequence[BaseScenarioFilterer]] = None,
        postfilters: Optional[Sequence[BaseScenarioFilterer]] = None,
        inverse_filters: bool = False,
        raise_exceptions: bool = True,
        max_scenarios: Optional[int] = None,
        seed: Optional[int] = None,
        skip_scenarios: Optional[Set[str]] = None
    ) -> None:
        """Init new ScenarioIterator instance 

        Args:
            directory (str): Base directory to load scenarios from.
            loop (bool, optional): Whether to recycle data. Defaults to False.
            skip_subvariants (bool): Filter out scenario subvariants (i.e. same road network).
            verbose (int): Verbosity level.
            shuffle (bool): Whether to randomly shuffle scenarios. Defaults to False.
            save_scenario_pickles (bool): Whether to export scenarios as pickles once they are loaded, allowing them to be loaded quicker in the future.
            load_scenario_pickles (bool): Whether to attempt to load previously exported scenarios from pickles.
            preprocessors (Optional[T_ScenarioPreprocessorsInput]): Optional list of preprocessors for preprocessing scenarios.
                If a tuple is specified for a preprocessor entry, the second tuple element.
                specifies how many separate preprocessing pipelines to create with the current preprocessor.
            raise_exceptions (bool) Whether to ignore exceptions from loading scenarios files and processing.
            max_scenarios (Optional[int]) Optional upper limit of scenarios that the iterator will yield.
            seed (Optional[int]) Random seed for scenario shuffling.
        """
        directory = Path(directory).resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory.absolute()} does not exist! Make sure that the specified path leads to a scenario file or a dictionary.")
        if directory.is_file() and directory.suffix == ".xml":
            self._scenario_files = [directory]
        else:
            self._scenario_files = list(map(Path, find_scenario_files(
                str(directory),
                skip_subvariants=skip_subvariants,
                skip_scenarios=skip_scenarios
            )))
        if len(self._scenario_files) == 0:
            raise FileNotFoundError(f"Directory {directory.absolute()} contains no scenario files!")

        self._scenario_file_iterator: Iterator[Path]
        self._current_preprocessor_pipeline_iterator: Optional[Iterator[T_ScenarioPreprocessorsPipeline]]
        self._current_preprocessor_pipeline_index: int = 0
        self._current_input_scenario_bundle: Optional[ScenarioBundle]
        self._directory = directory
        self._verbose = verbose
        self._loop = loop
        self._done = False
        self._raise_exceptions = raise_exceptions
        self._inverse_filters = inverse_filters
        self._save_pickles = save_scenario_pickles
        self._load_pickles = load_scenario_pickles
        self._max_scenarios = max_scenarios
        self._rng = Random(seed)

        if shuffle:
            self.shuffle(reset=False)

        preprocessors = BaseScenarioPreprocessor.cast(preprocessors or [])
        self._preprocessor_pipelines = ScenarioIterator.get_preprocessor_pipeline_combinations(preprocessors)
        self._prefilters = prefilters
        self._postfilters = postfilters
        self._output_scenarios_per_scenario = max(1, self.num_preprocessing_pipelines)
        self._length = len(self._scenario_files) * self._output_scenarios_per_scenario
        if self._max_scenarios is not None:
            self._length = min(self._length, self._max_scenarios)
        self._trajectory_pickle_directory = os.path.abspath(os.path.join(directory, os.pardir, "trajectories"))

        self._reset()

    def shuffle(self, seed: Optional[int] = None, reset: bool = True) -> None:
        if seed is not None:
            self._rng.seed(seed)
        self._rng.shuffle(self._scenario_files)
        if reset:
            self._reset()

    def _reset(self) -> None:
        self._scenario_file_iterator = (f for f in self._scenario_files)
        self._input_scenario_counter = 0
        self._output_scenario_counter_internal = 0
        self._output_scenario_counter = 0
        self._cycle = -1
        self._on_scenario_completed()

    @property
    def cycle(self) -> int:
        """Current iteration cycle"""
        return self._cycle

    @property
    def input_scenario_counter(self) -> int:
        return self._input_scenario_counter

    @property
    def output_scenario_counter(self) -> int:
        return self._output_scenario_counter

    @property
    def num_preprocessing_pipelines(self) -> int:
        return len(self._preprocessor_pipelines)

    @property
    def output_scenarios_per_scenario(self) -> int:
        return self._output_scenarios_per_scenario

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> ScenarioIterator:
        return self

    def __next__(self) -> ScenarioBundle:
        """Yields next data instance"""

        if self._done:
            raise StopIteration()

        scenario_bundle = self._clone_current_scenario_bundle()

        if self.num_preprocessing_pipelines > 0:
            assert self._current_preprocessor_pipeline_iterator is not None

            while True:
                try:
                    preprocessor_pipeline = next(self._current_preprocessor_pipeline_iterator)
                except StopIteration:
                    self._on_scenario_completed()
                    if self._current_input_scenario_bundle is None:
                        raise StopIteration()
                    scenario_bundle = self._clone_current_scenario_bundle()
                    preprocessor_pipeline = next(self._current_preprocessor_pipeline_iterator)
                assert self._current_input_scenario_bundle is not None
                self._current_input_scenario_bundle.preprocessed_scenario = deepcopy(self._current_input_scenario_bundle.input_scenario)
                scenario = self._current_input_scenario_bundle.preprocessed_scenario
                planning_problem_set = self._current_input_scenario_bundle.preprocessed_planning_problem_set
                accepted = True

                for preprocessor in preprocessor_pipeline:
                    if self._raise_exceptions:
                        scenario, planning_problem_set = preprocessor(scenario, planning_problem_set)
                    else:
                        try:
                            scenario, planning_problem_set = preprocessor(scenario, planning_problem_set)
                        except Exception as e:
                            logger.error(e, exc_info=True)
                            accepted = False
                            break

                if not accepted:
                    continue

                if self._postfilters is not None and len(self._postfilters) > 0:
                    for filter in self._postfilters:
                        filter_result = filter.filter_scenario(scenario, raise_exceptions=self._raise_exceptions)
                        accepted = (filter_result and not self._inverse_filters) or (not filter_result and self._inverse_filters)
                        if not accepted:
                            break

                if accepted:
                    break

            scenario_bundle.preprocessed_scenario = scenario
            scenario_bundle.preprocessed_planning_problem_set = planning_problem_set
        else:
            self._on_scenario_completed()

        self._current_preprocessor_pipeline_index += 1
        self._output_scenario_counter += 1
        self._output_scenario_counter_internal += 1

        if self._verbose >= 1:
            source = 'pickle' if scenario_bundle.load_scenario_from_pickle else 'xml'
            msg = f"Yielding {source} scenario {self._input_scenario_counter}: {scenario_bundle.preprocessed_scenario.scenario_id} ({self._current_preprocessor_pipeline_index}/{self.output_scenarios_per_scenario})"
            if self._verbose == 1:
                stdout(msg)
            else:
                logger.debug(msg)

        return scenario_bundle

    def _clone_current_scenario_bundle(self) -> ScenarioBundle:
        assert self._current_input_scenario_bundle is not None
        scenario_bundle = ScenarioBundle(
            preprocessed_scenario=self._current_input_scenario_bundle.input_scenario,
            input_scenario=self._current_input_scenario_bundle.input_scenario,
            input_scenario_file=self._current_input_scenario_bundle.input_scenario_file,
            preprocessed_planning_problem_set=self._current_input_scenario_bundle.planning_problem_set,
            planning_problem_set=self._current_input_scenario_bundle.planning_problem_set,
            trajectory_pickle_file=self._current_input_scenario_bundle.trajectory_pickle_file,
            load_scenario_from_pickle=self._current_input_scenario_bundle.load_scenario_from_pickle,
            internal_index=self._current_input_scenario_bundle.internal_index
        )
        return scenario_bundle

    def _on_scenario_completed(self) -> None:
        self._current_input_scenario_bundle = self._load_next_input_scenario()
        self._current_preprocessor_pipeline_iterator = iter(self._preprocessor_pipelines)
        self._current_preprocessor_pipeline_index = 0

    def _load_next_input_scenario(self) -> Optional[ScenarioBundle]:
        scenario_bundle: Optional[ScenarioBundle] = None
        while True:

            if self._max_scenarios is not None and self._output_scenario_counter_internal >= self._max_scenarios:
                if not self._loop and self._scenario_files:
                    self._done = True
                    return None
                self._scenario_file_iterator = (f for f in self._scenario_files)
                scenario_file = next(self._scenario_file_iterator)
            else:
                try:
                    scenario_file = next(self._scenario_file_iterator)
                except StopIteration:
                    if not self._loop and self._scenario_files:
                        self._done = True
                        return None
                    self._scenario_file_iterator = (f for f in self._scenario_files)
                    scenario_file = next(self._scenario_file_iterator)
            
            try:
                loaded_pickle = False
                pickle_export_file = scenario_file.with_suffix('.pickle_export.pkl')
                if self._load_pickles:
                    if pickle_export_file.is_file():
                        try:
                            scenario_bundle = cast(ScenarioBundle, load_pickle(str(pickle_export_file)))
                            if scenario_bundle.__dataclass_fields__.keys() == scenario_bundle.__dict__.keys():
                                scenario_bundle.load_scenario_from_pickle = True
                                loaded_pickle = True
                        except Exception as e:
                            logger.warn(f"Failed to load scenario pickle file '{pickle_export_file}': {repr(e)}. Reloading and recompressing scenario file.")
                            pass
                if not loaded_pickle:
                    file_reader = CommonRoadFileReader(str(scenario_file))
                    scenario, planning_problem_set = file_reader.open()
                    input_scenario, input_planning_problem_set = file_reader.open()

                    full_path_with_pickle_filename = scenario_file.with_suffix('.pkl')
                    just_pickle_filename = os.path.basename(full_path_with_pickle_filename)
                    trajectory_pickle_file = os.path.join(self._trajectory_pickle_directory, just_pickle_filename)

                    scenario_bundle = ScenarioBundle(
                        preprocessed_scenario=scenario,
                        input_scenario=input_scenario,
                        input_scenario_file=scenario_file,
                        load_scenario_from_pickle=False,
                        planning_problem_set=input_planning_problem_set,
                        preprocessed_planning_problem_set=planning_problem_set,
                        trajectory_pickle_file=Path(trajectory_pickle_file) if os.path.isfile(trajectory_pickle_file) else None,
                        internal_index=self._current_preprocessor_pipeline_index
                    )
                    if self._save_pickles:
                        save_pickle(scenario_bundle, file_path=str(pickle_export_file))
                
                assert scenario_bundle is not None

                accepted = True
                if self._prefilters is not None:
                    for filter in self._prefilters:
                        filter_result = filter.filter_scenario(scenario_bundle.input_scenario, raise_exceptions=self._raise_exceptions)
                        accepted = (filter_result and not self._inverse_filters) or (not filter_result and self._inverse_filters)
                        if not accepted:
                            break
                
                if accepted:
                    break

            except Exception as e:
                if self._raise_exceptions:
                    raise ScenarioIterationError(f"{scenario_file}") from e
                logger.warn(f"Failed to load scenario file '{scenario_file}': {repr(e)}")
            finally:
                self._input_scenario_counter += 1

        assert scenario_bundle is not None
        return scenario_bundle

    @staticmethod
    def get_preprocessor_pipeline_combinations(
        preprocessors: Sequence[Tuple[BaseScenarioPreprocessor, int]],
    ) -> List[T_ScenarioPreprocessorsPipeline]:
        combinations: List[T_ScenarioPreprocessorsPipeline] = []
        for element_index, (preprocessor, repeat_count) in enumerate(preprocessors):
            if element_index == 0:
                combinations = repeat_count * [[preprocessor]]
            else:
                new_combinations: List[T_ScenarioPreprocessorsPipeline] = []
                for intermediate_pipeline in combinations:
                    for _ in range(repeat_count):
                        new_combinations.append(intermediate_pipeline + [preprocessor])
                combinations = new_combinations
        return combinations
