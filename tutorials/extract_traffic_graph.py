import sys, os; sys.path.insert(0, os.getcwd())

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation

if __name__ == '__main__':
    input_scenario = 'data/osm_recordings/DEU_Munich-1_114_0_time_steps_1000_V1_0.xml'
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
