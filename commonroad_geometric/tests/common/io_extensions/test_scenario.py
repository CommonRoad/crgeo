from commonroad_geometric.common.io_extensions.hash import hash_scenario


def test_hash_scenario_deterministic(usa_us101_scenario_xml):
    hash1 = hash_scenario(usa_us101_scenario_xml)
    hash2 = hash_scenario(usa_us101_scenario_xml)
    assert hash1 == hash2


def test_hash_scenario_changes_on_change(usa_us101_scenario_xml):
    old_hash = hash_scenario(usa_us101_scenario_xml)
    usa_us101_scenario_xml.author = "This test"
    new_hash = hash_scenario(usa_us101_scenario_xml)
    assert old_hash != new_hash
