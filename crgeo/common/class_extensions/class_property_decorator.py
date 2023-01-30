from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Generic

from crgeo.common.types import T_AnyReturn


class ClassPropertyDescriptor(Generic[T_AnyReturn]):

    def __init__(self, fget: Any, fset: Any = None) -> None:
        self.fget = fget
        self.fset = fset

    @lru_cache(maxsize=None)
    def __get__(self, obj: Any, klass: Any = None) -> Any:
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj: Any, value: Any) -> Any:
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func: Any) -> ClassPropertyDescriptor:
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func: Callable[..., T_AnyReturn]) -> T_AnyReturn:
    if not isinstance(func, (classmethod, staticmethod)):
        return ClassPropertyDescriptor(classmethod(func))
    return ClassPropertyDescriptor(func)
