from abc import ABC, abstractmethod
from typing import Any, Callable, List
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin

from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D
from commonroad_geometric.rendering.types import RenderParams


T_OutputTransform = Callable[[RenderParams], RenderParams]

class BaseRendererPlugin(ABC, AutoReprMixin, StringResolverMixin):
    """
    Base class for renderer plugins, i.e. customizable methods for 
    layer-based drawing on the TrafficSceneRenderer canvas. 
    """
    def __init__(
        self,
        transforms: List[T_OutputTransform] = None,
        *args,
        **kwargs,
    ) -> None:
        ...
        
    @abstractmethod
    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams,
    ) -> None:
        ...
