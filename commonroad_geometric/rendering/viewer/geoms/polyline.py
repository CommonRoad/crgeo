from abc import ABC, abstractmethod


class PolyLine(ABC):

    @property
    @abstractmethod
    def vertices(self):
        ...

    @property
    @abstractmethod
    def start(self):
        ...

    @property
    @abstractmethod
    def end(self):
        ...
