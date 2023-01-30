from typing import Optional
from pathlib import Path
import logging
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.commonroad_dataset import CommonRoadDataset
from crgeo.dataset.transformation.base_transformation import BaseDataTransformation
from crgeo.dataset.transformation.implementations.feature_normalization.normalize_features import FeatureNormalizer, normalize_features, get_normalization_params_file_path

class FeatureNormalizationTransformation(BaseDataTransformation):
    def __init__(
        self,
        params_file_path: Path,
        max_fit_samples: Optional[int] = None
    ) -> None:
        self.params_file_path = params_file_path
        self.max_fit_samples = max_fit_samples
        self.feature_normalizer = FeatureNormalizer()
        try:
            self.feature_normalizer.load_parameters(params_file_path)
        except FileNotFoundError:
            pass

    def _transform_dataset(self, dataset: CommonRoadDataset) -> CommonRoadDataset:
        self.feature_normalizer = normalize_features(dataset=dataset, max_fit_samples=self.max_fit_samples)
        return dataset
    
    def _transform_data(self, data: CommonRoadData) -> CommonRoadData:
        assert self.feature_normalizer.is_fitted
        return self.feature_normalizer.normalize_features_(data) 

class FeatureUnnormalizationTransformation(FeatureNormalizationTransformation):
    def _transform_dataset(self, dataset: CommonRoadDataset) -> CommonRoadDataset:
        raise NotImplementedError("Cannot unnormalize dataset")
    
    def _transform_data(self, data: CommonRoadData) -> CommonRoadData:
        assert self.feature_normalizer.is_fitted
        return self.feature_normalizer.unnormalize_features_(data)
