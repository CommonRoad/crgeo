from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.transformation.dataset_transformation import dataset_transformation
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor, T_DataPostprocessorCallable
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


class BaseDataTransformation(ABC, AutoReprMixin, StringResolverMixin):

    """
    Base class for applying transformations to CommonRoadDatasets such as feature normalization.
    """

    def transform_dataset(self, dataset: CommonRoadDataset) -> CommonRoadDataset:
        if self.is_touched(dataset[0]):
            pass # TODO raise ValueError(f"Dataset has already been processed by {type(self).__name__}")
        dataset = self._transform_dataset(dataset)
        self.touch_dataset(dataset)
        return dataset

    @abstractmethod
    def _transform_dataset(self, dataset: CommonRoadDataset) -> CommonRoadDataset:
        ...

    def transform_data(self, data: CommonRoadData) -> CommonRoadData:
        if self.is_touched(data):
            raise ValueError(f"Data has already been processed by {type(self).__name__}")
        data = self._transform_data(data)
        self.touch_data(data)
        return data

    @abstractmethod
    def _transform_data(self, data: CommonRoadData) -> CommonRoadData:
        ...

    @property
    def touch_signature(self) -> str:
        return f'transformed_{type(self).__name__}'

    def is_touched(self, data: CommonRoadData) -> bool:
        return data._global_store.get(self.touch_signature, False)

    def touch_dataset(self, dataset: CommonRoadDataset) -> None:
        def transform(
            scenario_index: int,
            sample_index: int,
            data: CommonRoadData,
        ):
            self.touch_data(data)
            yield data
        dataset_transformation(
            dataset=dataset,
            transform=transform
        )

    def touch_data(self, data: CommonRoadData) -> None:
        setattr(data, self.touch_signature, True)

    def to_post_processor(self) -> T_DataPostprocessorCallable:
        parent = self

        def post_processor_callable(
            samples: List[CommonRoadData],
            simulation: Optional[BaseSimulation] = None,
            ego_vehicle: Optional[EgoVehicle] = None
        ) -> List[CommonRoadData]:
            return [parent.transform_data(data) for data in samples]
        return post_processor_callable
