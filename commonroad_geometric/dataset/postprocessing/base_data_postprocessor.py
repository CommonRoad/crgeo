from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, Union

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle

class T_DataPostprocessorCallable(Protocol):
    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        ...


T_LikeBaseDataPostprocessor = Union['BaseDataPostprocessor', T_DataPostprocessorCallable]

class BaseDataPostprocessor(ABC, AutoReprMixin, StringResolverMixin):

    """
    Base class for postprocessing CommonRoadData.
    As the final part of data preparation pipeline(preprocess -> dataset collector -> post-process), 
    Post-processors deal with the collected CommonRoadData, enabling computation or addition of extra attributes for CommonRoadData.
    In the case of RL training, Post-processors can also get access to simulation information, in which the CommonRoadData instances are collected from.

    Current Post-processors can be easily extended by writing another child class of BaseScenarioPostprocessor and overwriting the abstractmethod "__call__"

    ********************************************Design mode suggestion********************************************
    
    For additional features that are intended to be added to GNN, Feature computer is recommended, 
    which computes features in x Tensor for VirtualAttributesNodeStorage (vehicle or lanelet), 
    or in edge_attr Tensor for VirtualAttributesEdgeStorage (vehicle to vehicle, vehicle to lanelet, lanelet to lanelet).

    For computation of supportive attributes or dataprocessing based on collected CommonRoadData (e.g. discretization), 
    Post-processor is recommended.
    """

    @abstractmethod
    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        """
        Executes post-processing routine on data.

        Args:
            samples (List[CommonRoadData]): List of CommonRoadData data samples to be processed.

        Returns:
            List[CommonRoadData]: List of post-processed data instances.
        """
