from typing import Type
from commonroad_geometric.common.config import Config
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory

from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.geometric.project.base_geometric_project import BaseGeometricProject
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperimentConfig
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from commonroad_geometric.learning.base_project import register_run_command

from projects.geometric_models.dummy.model import DummyModel


class DummyProject(BaseGeometricProject):

    def configure_experiment(self, cfg: Config) -> GeometricExperimentConfig:
        return GeometricExperimentConfig(
            extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=cfg["dist_threshold_v2v"])
                ),
            ),
            data_collector_cls=ScenarioDatasetCollector,
            preprocessors=[],
            postprocessors=[],
            simulation_options=ScenarioSimulationOptions(
                lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_SHAPE,
                collision_checking=False
            )
        )

    def configure_model(self, cfg: Config, experiment: GeometricExperiment) -> Type[BaseGeometric]:
        return DummyModel

    @register_run_command
    def custom(self) -> None:
        print("You can register custom run modes like this")
