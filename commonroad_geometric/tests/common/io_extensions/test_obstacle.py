import pytest

from commonroad_geometric.common.io_extensions.hash import hash_dynamic_obstacle


@pytest.fixture(scope="function")
def dynamic_obstacle(usa_us101_scenario_xml):
    return usa_us101_scenario_xml.dynamic_obstacles[0]


def test_hash_dynamic_obstacle_deterministic(dynamic_obstacle):
    hash1 = hash_dynamic_obstacle(dynamic_obstacle)
    hash2 = hash_dynamic_obstacle(dynamic_obstacle)
    assert hash1 == hash2


def test_hash_dynamic_obstacle_changes_on_change(dynamic_obstacle):
    old_hash = hash_dynamic_obstacle(dynamic_obstacle)
    dynamic_obstacle.initial_center_lanelet_ids.add(1)
    new_hash = hash_dynamic_obstacle(dynamic_obstacle)
    assert old_hash != new_hash
