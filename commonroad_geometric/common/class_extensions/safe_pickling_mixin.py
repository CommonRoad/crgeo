from typing import Dict, Set, Any
from commonroad_geometric.common.utils.filesystem import is_picklable


class SafePicklingMixin:
    """
    Mixin class that ensures that a class can be successfully dumped as a pickle object
    by ignoring non-picklable attributes.
    """

    def __getstate__(self) -> Dict[str, Any]:
        if not hasattr(self, '__pickle_exports__'):
            self.__pickle_exports__: Set[str] = {k for k, v in self.__dict__.items() if is_picklable(v)}
        export = {k: self.__dict__[k] for k in self.__pickle_exports__}
        return export
