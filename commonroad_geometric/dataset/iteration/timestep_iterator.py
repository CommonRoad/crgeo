from __future__ import annotations

from typing import Union

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.io_extensions.scenario import get_scenario_timestep_bounds


class TimeStepIterator(AutoReprMixin):
    """Class for iterating over time-steps in a CommonRoad scenario.
    Example usage:
    .. code-block:: python

        timestep_iterator = TimeStepIterator(
            scenario=scenario
        )
        for time_step in timestep_iterator:
            print(time_step)
    """

    def __init__(
        self,
        scenario: Union[str, Scenario],
        loop: bool = False,
    ) -> None:
        """Init new TimeStepIterator instance 

        Args:
            scenario (Union[str, Scenario]): Scenario to iterate over.
            loop (bool): Whether to iterate over the scenario in a loop. Defaults to False.
        """

        if isinstance(scenario, str):
            scenario, _ = CommonRoadFileReader(filename=scenario).open()
        self._scenario = scenario

        self._initial_time_step, self._final_time_step = get_scenario_timestep_bounds(self._scenario)

        self._iterator = (t for t in range(self._initial_time_step, self._final_time_step))
        self._cycle = 0
        self._loop = loop

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def cycle(self) -> int:
        """Current iteration cycle"""
        return self._cycle

    def __iter__(self) -> TimeStepIterator:
        return self

    def __next__(self) -> int:
        """Yields next time-step"""
        try:
            time_step = next(self._iterator)
        except StopIteration:
            if not self._loop:
                raise StopIteration()
            self._iterator = (t for t in range(self._initial_time_step, self._final_time_step))
            self._cycle += 1
            time_step = next(self._iterator)
        return time_step
