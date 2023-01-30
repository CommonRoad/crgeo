import unittest
import numpy as np

from crgeo.common.io_extensions.mock_objects import create_dummy_scenario
from crgeo.common.io_extensions.lanelet_network import segment_lanelets
from commonroad.common.file_reader import CommonRoadFileReader


class TestLaneletNetwork(unittest.TestCase):

    def test_segment_lanelets_deterministic(self):
        
        scenario, planning_problem_set = CommonRoadFileReader('./crgeo/tests/io_extensions/USA_US101-26_1_T-1.xml').open()

        network_1 = segment_lanelets(scenario, 5.0)
        network_2 = segment_lanelets(scenario, 5.0)

        self.assertTrue(len(network_1.lanelets) == len(network_2.lanelets))
        for l in range(len(network_1.lanelets)):
            lanelet_1 = network_1.lanelets[l]
            lanelet_2 = network_2.lanelets[l]
            self.assertTrue(lanelet_1.lanelet_id == lanelet_2.lanelet_id)
            self.assertTrue(np.allclose(lanelet_1.center_vertices, lanelet_2.center_vertices))
        assert len(network_1.lanelets)

if __name__ == "__main__":
    unittest.main()
