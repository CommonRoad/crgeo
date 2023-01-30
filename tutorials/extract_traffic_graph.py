import sys, os; sys.path.insert(0, os.getcwd())

from commonroad.common.file_reader import CommonRoadFileReader

from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from crgeo.simulation.interfaces.static.scenario_simulation import ScenarioSimulation

if __name__ == '__main__':
    input_scenario = 'data/other/ARG_Carcarana-1_7_T-1.xml'
    scenario, _ = CommonRoadFileReader(input_scenario).open()
    traffic_extractor = TrafficExtractor(
        simulation=ScenarioSimulation(initial_scenario=input_scenario),
        options=TrafficExtractorOptions(
            edge_drawer=VoronoiEdgeDrawer()
        )
    )
    traffic_extractor.simulation.start()
    for data in traffic_extractor:
        print(data)
