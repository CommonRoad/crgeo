from __future__ import annotations

from abc import ABC
from typing import Generic, TYPE_CHECKING, Type

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.dataset.extraction.base_extractor import T_BaseExtractor, T_BaseExtractorOptions

if TYPE_CHECKING:
    from commonroad_geometric.simulation.base_simulation import BaseSimulation


class BaseExtractorFactory(ABC, Generic[T_BaseExtractor, T_BaseExtractorOptions], AutoReprMixin):
    def __init__(
        self,
        extractor_cls: Type[T_BaseExtractor],
        options: T_BaseExtractorOptions
    ) -> None:
        self._extractor_cls = extractor_cls
        self._options = options

    def __call__(
        self,
        simulation: BaseSimulation
    ) -> T_BaseExtractor:
        return self._extractor_cls(
            simulation=simulation,
            options=self._options
        )
