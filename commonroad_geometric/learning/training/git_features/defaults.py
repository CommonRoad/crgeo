from typing import Callable, Dict

from commonroad_geometric.learning.training.git_features.base_git_feature_computer import BaseGitFeatureParams
from commonroad_geometric.learning.training.git_features.implementations.get_sha import GetSha
from commonroad_geometric.learning.training.git_features.types import Git_Metadata

get_branch: Callable[[BaseGitFeatureParams], Dict] = lambda params: {
    Git_Metadata.Branch.value: params.repo.active_branch.name}
get_author: Callable[[BaseGitFeatureParams], Dict] = lambda params: {
    Git_Metadata.Author.value: params.repo.head.object.author.name}


DEFAULT_GIT_FEATURE_COLLECTORS = [
    GetSha(),
    get_branch,
    get_author,
]
