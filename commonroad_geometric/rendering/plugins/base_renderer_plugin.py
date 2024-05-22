from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer

T_RendererPlugin = TypeVar("T_RendererPlugin", bound="BaseRendererPlugin")


@dataclass
class BaseRenderPlugin(ABC, AutoReprMixin, StringResolverMixin):
    """
    Base class for render plugins, i.e. customizable methods for layer-based drawing on the TrafficSceneRenderer
    canvas. Implemented as a dataclass to handle all kinds of style attributes.
    """

    @abstractmethod
    def render(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        ...

    def __call__(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        r"""
        First automagically overwrites the attributes of the renderer plugin with any additional kwargs given in
        params.render_kwargs if the name of the attribute matches that of the kwarg. Then calls render().
        Finally, restores the overwritten attributes of the renderer plugin to their prior state.

        Args:
            viewer (T_Viewer): The viewer to render to.
            params (RenderParams): The parameters to render.
        """
        attribute_name_to_old_value = self._overwrite_attributes_with_kwargs(kwargs=params.render_kwargs)
        self.render(
            viewer=viewer,
            params=params
        )
        if attribute_name_to_old_value:
            self._restore_attributes(attribute_name_to_old_value=attribute_name_to_old_value)

    def _overwrite_attributes_with_kwargs(
        self,
        kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        r"""
        Overwrites attributes of the renderer plugin with the given kwargs if their names match.

        Args:
            kwargs (dict[str, Any]): Arbitrary key-word arguments given to the render pipeline.

        Returns:
            Dictionary of overwritten key-word arguments mapped to their old/prior values
        """
        attribute_name_to_old_value = {}
        for attribute_name, new_value in kwargs.items():
            if attribute_name in self.__dict__:
                old_value = self.__dict__[attribute_name]
                attribute_name_to_old_value[attribute_name] = old_value
                setattr(self, attribute_name, new_value)
        return attribute_name_to_old_value

    def _restore_attributes(
        self,
        attribute_name_to_old_value: dict[str, Any]
    ) -> None:
        r"""
        Restores overwritten attributes of the renderer plugin to its prior/old state.

        Args:
            attribute_name_to_old_value (dict[str, Any]): Dictionary of overwritten key-word arguments mapped to
                                                          their old/prior values
        """
        for attribute_name, old_value in attribute_name_to_old_value.items():
            setattr(self, attribute_name, old_value)
