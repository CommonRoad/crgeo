from __future__ import annotations

from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.extraction.road_network.road_network_extractor import RoadNetworkExtractor, RoadNetworkExtractorOptions


class RoadNetworkExtractorFactory(BaseExtractorFactory[RoadNetworkExtractor, RoadNetworkExtractorOptions]):
    def __init__(self, options: RoadNetworkExtractorOptions) -> None:
        super(RoadNetworkExtractorFactory, self).__init__(
            extractor_cls=RoadNetworkExtractor,
            options=options
        )
