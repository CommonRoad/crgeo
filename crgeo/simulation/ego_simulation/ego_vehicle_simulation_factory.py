from __future__ import annotations

from typing import Type

from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from crgeo.dataset.iteration.scenario_iterator import ScenarioBundle
from crgeo.simulation.base_simulation import BaseSimulation, BaseSimulationOptions
from crgeo.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from crgeo.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation, EgoVehicleSimulationOptions
from crgeo.simulation.ego_simulation.respawning import BaseRespawner, BaseRespawnerOptions
from crgeo.simulation.interfaces.static.compressed_scenario_simulation import CompressedSimulationOptions


class EgoVehicleSimulationFactory(AutoReprMixin):
    def __init__(
        self,
        simulation_cls: Type[BaseSimulation],
        simulation_options: BaseSimulationOptions,
        control_space_cls: Type[BaseControlSpace],
        control_space_options: BaseControlSpaceOptions,
        respawner_cls: Type[BaseRespawner],
        respawner_options: BaseRespawnerOptions,
        ego_vehicle_simulation_options: EgoVehicleSimulationOptions,
        extractor_factory: TrafficExtractorFactory,
    ) -> None:
        self._simulation_cls = simulation_cls
        self._simulation_options = simulation_options
        self._control_space_cls = control_space_cls
        self._control_space_options = control_space_options
        self._respawner_cls = respawner_cls
        self._respawner_options = respawner_options
        self._ego_vehicle_simulation_options = ego_vehicle_simulation_options
        self._extractor_factory = extractor_factory

    def create(
        self,
        scenario_bundle: ScenarioBundle,
    ) -> EgoVehicleSimulation:
        if scenario_bundle.trajectory_pickle_file is not None and isinstance(self._simulation_options, CompressedSimulationOptions):
            self._simulation_options.trajectory_pickle_file = scenario_bundle.trajectory_pickle_file

        simulation = self._simulation_cls(
            initial_scenario=scenario_bundle.preprocessed_scenario,
            options=self._simulation_options,
        )

        extractor = self._extractor_factory.create(simulation=simulation)
        ego_vehicle_simulation = EgoVehicleSimulation(
            simulation=simulation,
            control_space=self._control_space_cls(options=self._control_space_options),
            respawner=self._respawner_cls(options=self._respawner_options),
            traffic_extractor=extractor,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            options=self._ego_vehicle_simulation_options
        )
        return ego_vehicle_simulation
