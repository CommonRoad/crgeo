from __future__ import annotations

from abc import ABC
from typing import Generic, TYPE_CHECKING, Type, TypeVar

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.dataset.extraction.base_extractor import TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions

if TYPE_CHECKING:
    from commonroad_geometric.simulation.base_simulation import BaseSimulation

TypeVar_BaseExtractorFactory = TypeVar('TypeVar_BaseExtractorFactory', bound='BaseExtractorFactory')


class BaseExtractorFactory(ABC, Generic[TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions], AutoReprMixin):
    def __init__(self, extractor_cls: Type[TypeVar_BaseExtractor], options: TypeVar_BaseExtractorOptions) -> None:
        self._extractor_cls = extractor_cls
        self._options = options

    def create(self, simulation: BaseSimulation) -> TypeVar_BaseExtractor:
        return self._extractor_cls(
            simulation=simulation,
            options=self._options
        )
