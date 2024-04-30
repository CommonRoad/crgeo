from pathlib import Path
from typing import Union

from commonroad.scenario.scenario import Scenario
from typing_extensions import assert_never

from commonroad_geometric.simulation.base_simulation import BaseSimulation, T_SimulationOptions
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from commonroad_geometric.simulation.interfaces.static.unpopulated_simulation import UnpopulatedSimulation, UnpopulatedSimulationOptions


class SimulationFactory:
    def __init__(
        self,
        options: T_SimulationOptions
    ) -> None:
        r"""
        Args:
            options (T_SimulationOptions): Options for the simulation.
        """
        self.options = options

    def __call__(
        self,
        initial_scenario: Union[Scenario, Path],
    ) -> BaseSimulation:
        r"""
        Args:
            initial_scenario (Union[Scenario, Path]): Scenario for which simulation should be created.

        Returns:
            Simulation for scenario.
        """
        match self.options:
            case SumoSimulationOptions() as options:
                return SumoSimulation(
                    initial_scenario=initial_scenario,
                    options=options
                )
            case ScenarioSimulationOptions() as options:
                return ScenarioSimulation(
                    initial_scenario=initial_scenario,
                    options=options
                )
            case UnpopulatedSimulationOptions() as options:
                return UnpopulatedSimulation(
                    initial_scenario=initial_scenario,
                    options=options
                )
            case _:
                assert_never(self.options)
