from __future__ import annotations

from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor, TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class TrafficExtractorFactory(BaseExtractorFactory[TrafficExtractor, TrafficExtractorOptions]):
    def __init__(self, options: TrafficExtractorOptions) -> None:
        super(TrafficExtractorFactory, self).__init__(
            extractor_cls=TrafficExtractor,
            options=options
        )


class TemporalTrafficExtractorFactory(BaseExtractorFactory[TemporalTrafficExtractor, TemporalTrafficExtractorOptions]):
    def __init__(
        self,
        options: TemporalTrafficExtractorOptions,
        traffic_extractor_factory: BaseExtractorFactory,
    ) -> None:
        super().__init__(
            extractor_cls=TemporalTrafficExtractor,
            options=options,
        )
        self._traffic_extractor_factory = traffic_extractor_factory

    def __call__(self, simulation: BaseSimulation) -> TemporalTrafficExtractor:
        traffic_extractor = self._traffic_extractor_factory(simulation=simulation)
        return self._extractor_cls(
            traffic_extractor=traffic_extractor,
            options=self._options,
        )
