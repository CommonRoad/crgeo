import inspect
from typing import Any, Dict
import json

from crgeo.common.utils.filesystem import SafeJsonEncoder

class AutoReprMixin:
    """
    Automatically generates __repr__ str for class.
    """

    def _get_repr_attributes(self) -> Dict[str, Any]:
        init_params = inspect.signature(self.__init__).parameters # type: ignore
        repr_attributes: Dict[str, Any] = {}
        for param in init_params:
            if param in self.__dict__:
                repr_attributes[param] = self.__dict__[param]
            elif '_' + param in self.__dict__:
                repr_attributes[param] = self.__dict__['_' + param]
        return repr_attributes

    def __repr__(self) -> str:
        repr_attributes = self._get_repr_attributes()
        repr_str = f"{type(self).__name__}({', '.join((k + '=' + str(v) for k, v in repr_attributes.items() if len(str(v)) < 40))})"
        return repr_str

    def __str__(self) -> str:
        repr_attributes = self._get_repr_attributes()
        repr_str = f"{type(self).__name__}({json.dumps(repr_attributes, indent=2, cls=SafeJsonEncoder)})"
        return repr_str

