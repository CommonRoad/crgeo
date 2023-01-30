from typing import Generic, Optional, Type, TypeVar

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.type_checking import GenericTypeChecker

T_CachedProperty = TypeVar("T_CachedProperty")


class EmptyCacheException(ValueError):
    def __init__(self) -> None:
        super().__init__("Cache is empty")


class CachedProperty(Generic[T_CachedProperty], AutoReprMixin):
    def __init__(
        self, 
        value_type: Type[T_CachedProperty],
        init_time_step: Optional[int] = None,
        init_value: Optional[T_CachedProperty] = None,
        type_checking: bool = False
    ) -> None:
        self._time_step: Optional[int] = init_time_step
        self._value: Optional[T_CachedProperty] = init_value
        self._type_checking = type_checking
        if type_checking:
            self._time_step_type_checker = GenericTypeChecker(int)
            self._time_step_type_checker.assert_type(init_time_step, accept_none=True)
            self._value_type_checker = GenericTypeChecker(value_type)
            self._value_type_checker.assert_type(init_value, accept_none=True)

    def set(self, time_step: int, value: T_CachedProperty, overwrite: bool = False) -> None:
        if self._type_checking:
            self._time_step_type_checker.assert_type(time_step)
            self._value_type_checker.assert_type(value)
        if self._time_step == time_step and not overwrite:
            return
        self._time_step = time_step
        self._value = value

    def is_settable(self, time_step: int, overwrite: bool = False) -> bool:
        if self._type_checking:
            self._time_step_type_checker.assert_type(time_step)
        return self._time_step != time_step or overwrite

    def get(self) -> T_CachedProperty:
        if self._value is None:
            raise EmptyCacheException()
        return self._value

    def clear(self) -> None:
        self._time_step = None
        self._value = None

    def clear_value(self) -> None:
        self._value = None

    @property
    def time_step(self) -> Optional[int]:
        return self._time_step

    @property
    def value(self) -> T_CachedProperty:
        return self.get()
