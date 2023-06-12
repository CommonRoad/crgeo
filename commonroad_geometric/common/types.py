
from typing import Union, Type, Generic, Protocol, TypeVar, Any
from sys import maxsize


class _UnlimitedMeta(type):
    __str__ = __repr__ = lambda self: "Unlimited"
    __int__ = lambda self: maxsize
    __gt__ = lambda self, other: True
    __ge__ = lambda self, other: True
    __lt__ = lambda self, other: False
    __le__ = lambda self, other: False
    __add__ = lambda self, other: maxsize
    __sub__ = lambda self, other: -maxsize

class Unlimited(metaclass=_UnlimitedMeta):
    pass


T_CountParam = Union[int, Type[Unlimited]]


T_AnyReturn = TypeVar('T_AnyReturn', covariant=True)

class AnyCallable(Protocol, Generic[T_AnyReturn]):
      def __call__(self, *args: Any, **kwargs: Any) -> T_AnyReturn: ...
