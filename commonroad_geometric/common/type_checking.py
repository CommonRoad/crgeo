from typing import Any, Generic, Iterable, Type, TypeVar

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin

T = TypeVar('T')
V = TypeVar("V")
sentinel = object()


class GenericTypeChecker(Generic[T], AutoReprMixin):
    def __init__(self, generic_arg: Type[T]) -> None:
        self._generic_arg = generic_arg

    def is_right_type(self, x: Any) -> bool:
        return isinstance(x, self._generic_arg)

    def assert_type(self, x: Any, accept_none: bool = False) -> None:
        if accept_none and x is None:
            return
        if not self.is_right_type(x):
            raise TypeError(f"Invalid type {type(x)} (expected {self._generic_arg.__name__})")  # type: ignore


def is_mutable(obj: Any) -> bool:
    try:
        return isinstance(obj, list) or hasattr(obj, '__dict__')
    except AttributeError:
        return False


def all_same(it: Iterable[V]) -> bool:
    val = sentinel
    for v in it:
        if val is sentinel:
            val = v
        elif v != val:
            return False
    return True
