from enum import Enum


class RewardLossMetric(Enum):
    L1 = 'l1'
    L2 = 'l2'
    Gaussian = 'gaussian'


class MissingFeatureException(ValueError):
    def __init__(self, feature: str):
        super().__init__(f"Missing feature '{feature}' required for reward computation")
