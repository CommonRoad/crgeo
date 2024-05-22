from sys import maxsize
from typing import Union, Type, Generic, Protocol, TypeVar, Any


class _UnlimitedMeta(type):
    __str__ = __repr__ = lambda self: "Unlimited"
    def __int__(self): return maxsize
    def __gt__(self, other): return True
    def __ge__(self, other): return True
    def __lt__(self, other): return False
    def __le__(self, other): return False
    def __add__(self, other): return maxsize
    def __sub__(self, other): return -maxsize


class Unlimited(metaclass=_UnlimitedMeta):
    pass


T_CountParam = Union[int, Type[Unlimited]]


T_AnyReturn = TypeVar('T_AnyReturn', covariant=True)


class AnyCallable(Protocol, Generic[T_AnyReturn]):
    def __call__(self, *args: Any, **kwargs: Any) -> T_AnyReturn: ...
