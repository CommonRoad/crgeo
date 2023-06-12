from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from git import Repo

TypeVar_GitParams = TypeVar(
    "TypeVar_GitParams",
    bound='BaseGitFeatureParams',
)


@dataclass
class BaseGitFeatureParams:
    repo: Repo


class BaseGitFeatureComputer(Generic[TypeVar_GitParams], ABC):
    """
    Base class for custom git feature computers for embedding desired values in a git feature
    dictionary
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(
        self,
        params: TypeVar_GitParams,
    ):
        ...
