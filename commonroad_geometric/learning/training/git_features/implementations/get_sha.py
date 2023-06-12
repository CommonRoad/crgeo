from typing import Dict

from commonroad_geometric.learning.training.git_features.base_git_feature_computer import BaseGitFeatureComputer, BaseGitFeatureParams
from commonroad_geometric.learning.training.git_features.types import Git_Metadata


class GetSha(BaseGitFeatureComputer[BaseGitFeatureParams]):
    def __init__(self):
        pass
    
    def __call__(
        self,
        params: BaseGitFeatureParams,
    ):

        sha = params.repo.head.object.hexsha
        features: Dict[str,] = {
            Git_Metadata.Sha.value: sha,
        }
        return features
