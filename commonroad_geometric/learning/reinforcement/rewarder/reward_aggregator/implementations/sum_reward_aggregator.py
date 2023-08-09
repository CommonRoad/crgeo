from typing import Dict

from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.base_reward_aggregator import BaseRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer


class SumRewardAggregator(BaseRewardAggregator):
    """
    Aggregator class for summing the reward components.
    """
    def _aggregate(self, reward_components: Dict[BaseRewardComputer, float]) -> float:
        return sum(reward_components.values())
