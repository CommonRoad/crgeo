import os
from typing import List
from commonroad.common.util import Interval
from dataclasses import dataclass, field

# TODO Unused


@dataclass
class SumoScenarioConfig():
    # scenario name and also folder name under which all scenario files are stored
    # sumo_commandline_config.net_file == f'{dirpath}/{scenario_name}.net.xml'
    # _cr_sumo_simulation.initialize(), _cr_sumo_simulation.traci_label(), _cr_sumo_simulation.commonroad_scenario_at_time_step(), _cr_sumo_simulation.commonroad_scenarios_all_time_steps(), sumo_scenario.init_from_scenario()
    scenario_name: str = 'ZAM_TMP_0-0'
    country_id: str = 'ZAM'                     # TODO: unused reference

    # ego vehicle
    # sumo_scenario.recreate_route_file(), sumo_scenario.init_from_cr_file(), sumo_scenario.init_from_net_file()
    ego_start_time: int = 0
    # sumo_scenario.init_from_net_file(), util.generate_rou_file()
    ego_ids: List[str] = field(default_factory=lambda: [])
    n_ego_vehicles: int = 0                         # sumo_scenario.init_from_net_file(), util.generate_rou_file()

    ##
    # TRAFFIC GENERATION
    ##
    # probability that vehicles will start at the fringe of the network (edges without
    # predecessor), and end at the fringe of the network (edges without successor).
    fringe_factor: int = 1000000000                            # external/sumocr/maps/util.generate_rou_file()
    # number of vehicle departures per second
    veh_per_second: int = 50                                    # external/sumocr/maps/util.generate_rou_file()
    # Interval of departure times for vehicles
    # TODO departure_interval_vehicles:    Interval(int, int) = Interval(0, 30)        # external/sumocr/maps/util.generate_rou_file()
    # max. number of vehicles in route file
    n_vehicles_max: int = 30                                    # sumo_scenario.init_from_net_file()
    # max. number of vehicles per km
    max_veh_per_km: int = 70                                    # external/sumocr/maps/util.generate_rou_file()
