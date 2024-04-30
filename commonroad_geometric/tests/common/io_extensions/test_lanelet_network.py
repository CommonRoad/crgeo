import numpy as np

from commonroad_geometric.common.io_extensions.lanelet_network import segment_lanelets


def test_segment_lanelets_deterministic_xml(usa_us101_scenario_xml):
    network_1 = segment_lanelets(usa_us101_scenario_xml, 5.0)
    network_2 = segment_lanelets(usa_us101_scenario_xml, 5.0)

    assert len(network_1.lanelets) == len(network_2.lanelets)

    for lanelet_1, lanelet_2 in zip(network_1.lanelets, network_2.lanelets, strict=True):
        assert lanelet_1.lanelet_id == lanelet_2.lanelet_id
        assert np.allclose(lanelet_1.center_vertices, lanelet_2.center_vertices)
    assert len(network_1.lanelets)
