from collections.abc import Hashable
from typing import Any, Dict, Tuple, TypeVar, Union

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.types import AnyCallable

RandomNumber = Union[int, float]
RandomObject = Union[RandomNumber, Tuple[RandomNumber, ...]]
T_CachedProperty = TypeVar("T_CachedProperty")


class CachedRNG(AutoReprMixin):
    def __init__(
        self, 
        rng: AnyCallable[float]
    ) -> None:
        self._rng = rng
        self._cache: Dict[Hashable, RandomObject] = {}

    def __call__(
        self, 
        key: Hashable,
        n: int = 1,
        **rng_kwargs: Any
    ) -> Any:
        if key not in self._cache:
            if n == 1:
                self._cache[key] = self._rng(**rng_kwargs)
            else:
                self._cache[key] = tuple((self._rng(**rng_kwargs) for _ in range(n)))
        return self._cache[key]
