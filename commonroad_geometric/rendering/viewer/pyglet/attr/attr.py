from abc import ABC, abstractmethod


class Attr(ABC):

    @abstractmethod
    def enable(self) -> None:
        ...

    @abstractmethod
    def disable(self) -> None:
        ...
