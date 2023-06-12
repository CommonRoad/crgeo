from functools import partial
from typing import Dict, List

import git

from commonroad_geometric.common.utils.functions import partial_satisfiable
from commonroad_geometric.learning.training.git_features.base_git_feature_computer import BaseGitFeatureComputer, BaseGitFeatureParams, TypeVar_GitParams
from commonroad_geometric.learning.training.git_features.types import Git_Metadata


class GitFeatureCollector:
    """ Gets data from git and sends it over to git feature computers to get a dictionary of relevant information
    """

    def __init__(
        self,
        feature_computers: List[BaseGitFeatureComputer[TypeVar_GitParams]]
    ) -> None:
        repo = git.Repo(search_parent_directories=True)
        self._feature_computers = feature_computers
        self._repo = repo

    def __call__(self) -> Dict:
        features: Dict[str,] = {Git_Metadata.__getitem__(i)._value_: None for i in Git_Metadata.__members__}
        for feature_computer in self._feature_computers:
            # Only required if we require certain initializations
            feature_callable = partial(feature_computer, params=BaseGitFeatureParams(repo=self._repo))
            if partial_satisfiable(feature_callable):
                feature_callable_features = feature_callable()
                for key in feature_callable_features:
                    features[key] = feature_callable_features[key]
        return features

    @property
    def get_repo(self) -> bool:
        return self._repo
