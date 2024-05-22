from functools import reduce

from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.identity_preprocessor import IdentityPreprocessor


def chain_preprocessors(*preprocessors: BaseScenarioPreprocessor) -> BaseScenarioPreprocessor:
    if not preprocessors:
        return IdentityPreprocessor()
    return reduce(lambda pp1, pp2: pp1 >> pp2, preprocessors)
