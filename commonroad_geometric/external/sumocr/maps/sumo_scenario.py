import logging
import os
import pathlib
import xml.etree.ElementTree as et

from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_geometric.external.sumocr.interface.util import NetError
from commonroad_geometric.external.sumocr.sumo_config import DefaultConfig
from commonroad_geometric.external.sumocr.maps.scenario_wrapper import AbstractScenarioWrapper
from commonroad_geometric.external.sumocr.maps.util import get_scenario_name_from_crfile, get_scenario_name_from_netfile, generate_rou_file

try:
    from sumo_map.cr2sumo.converter import CR2SumoMapConverter
    from sumo_map.config import SumoConfig
    from sumo_map.sumo2cr import convert_net_to_cr
    cr_map_converter_installed = True
except ImportError:
    cr_map_converter_installed = False

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2021.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class ScenarioWrapper(AbstractScenarioWrapper):
    def __init__(self):
        self.scenario_name: str = ''
        self.net_file: str = ''
        self.cr_map_file: str = ''
        self.sumo_cfg_file = None
        self.ego_start_time: int = 0
        self.sumo_net = None
        self.lanelet_network: LaneletNetwork = None
        self._route_planner = None

    def initialize(self,
                   scenario_name: str,
                   sumo_cfg_file: str,
                   cr_map_file: str,
                   ego_start_time: int = None) -> None:
        """
        Initializes the ScenarioWrapper.

        :param scenario_name: the name of the scenario
        :param sumo_cfg_file: the .sumocfg file
        :param cr_map_file: the commonroad map file
        :param ego_start_time: the start time of the ego vehicle

        """
        self.scenario_name = scenario_name
        self.sumo_cfg_file = sumo_cfg_file
        self.net_file = self._get_net_file(self.sumo_cfg_file)
        self.cr_map_file = cr_map_file
        self.ego_start_time = ego_start_time
        self.lanelet_network = CommonRoadFileReader(
            self.cr_map_file).open_lanelet_network()

    @classmethod
    def init_from_scenario(cls,
                           config: DefaultConfig,
                           scenario_path=str,
                           ego_start_time: int = None,
                           cr_map_file=None) -> 'ScenarioWrapper':
        """
        Initializes the ScenarioWrapper according to the given scenario_name/ego_start_time and returns the ScenarioWrapper.
        :param config: config file for the initialization, contain scenario_name.
        :param scenario_path: path to the scenario folder
        :param ego_start_time: the start time of the ego vehicle.
        :param cr_map_file: path to commonroad map, if not in scenario folder

        """
        assert isinstance(
            config,
            DefaultConfig), f'Expected type DefaultConfig, got {type(config)}'

        obj = cls()
        scenario_path = config.scenarios_path if config.scenarios_path is not None else scenario_path
        sumo_cfg_file = os.path.join(scenario_path,
                                     config.scenario_name + '.sumo.cfg')
        if cr_map_file is None:
            cr_map_file = os.path.join(scenario_path,
                                       config.scenario_name + '.cr.xml')

        obj.initialize(config.scenario_name, sumo_cfg_file, cr_map_file,
                       ego_start_time)
        return obj

    @classmethod
    def recreate_route_file(
            cls,
            sumo_cfg_file,
            conf: DefaultConfig = DefaultConfig) -> 'ScenarioWrapper':
        """
        Creates new .rou.xml file and returns ScenarioWrapper. Assumes .cr.xml, .net.xml and .sumo.cfg file have already been created in scenario folder.

        :param sumo_cfg_file:
        :param conf:

        """
        sumo_scenario = cls()
        out_folder = os.path.dirname(sumo_cfg_file)
        net_file = sumo_scenario._get_net_file(sumo_cfg_file)
        scenario_name = get_scenario_name_from_netfile(net_file)
        generate_rou_file(net_file, out_folder, conf)

        cr_map_file = os.path.join(os.path.dirname(__file__),
                                   '../../scenarios/', scenario_name,
                                   scenario_name + '.cr.xml')

        sumo_scenario.initialize(scenario_name, sumo_cfg_file, cr_map_file,
                                 conf.ego_start_time)
        return sumo_scenario

    @classmethod
    def init_from_cr_file(
        cls, cr_file: str,
        conf: DefaultConfig = DefaultConfig()) -> 'ScenarioWrapper':
        """
        Convert CommonRoad xml to sumo net file and return Scenario Wrapper.
        :param cr_file: path to the cr map file
        :param conf: configuration file
        :return:total length of all lanes, conversion_possible
        """
        if cr_map_converter_installed is False:
            raise EnvironmentError('This function requires map converter from the package'
                                   'crdesigner which is yet to be released.')

        scenario_name = get_scenario_name_from_crfile(cr_file)
        out_folder = os.path.join(conf.scenarios_path, scenario_name)
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        net_file = os.path.join(out_folder, scenario_name + '.net.xml')

        # convert from cr to net
        cr2sumo_converter = CR2SumoMapConverter.from_file(
            cr_file,
            SumoConfig.from_dict({
                attr_name: getattr(conf, attr_name)
                for attr_name in dir(conf) if not attr_name.startswith('_')
            }))
        # Write final net file
        logging.info('write map to path', net_file)
        conversion_possible = cr2sumo_converter.convert_to_net_file(out_folder)
        assert conversion_possible, "Conversion from cr file to net file failed!"
        sumo_scenario = cls()
        sumo_scenario.initialize(scenario_name, cr2sumo_converter.sumo_cfg_file, cr_file,
                                 conf.ego_start_time)
        return sumo_scenario

    @classmethod
    def init_from_net_file(
        cls,
        net_file: str,
        cr_map_path: str = None,
        conf: DefaultConfig = DefaultConfig()
    ) -> 'ScenarioWrapper':
        """
        Convert net file to CommonRoad xml and generate specific ego vehicle either by using generated vehicles and/or by initial states.

        :param net_file: path of .net.xml file
        :param cr_map_path: optionally specify commonroad map
        :param conf: configuration file for additional parameters

        """
        assert len(conf.ego_ids) <= conf.n_ego_vehicles, "total number of given ego_vehicles must be <= n_ego_vehicles, but {}not<={}"\
            .format(len(conf.ego_ids),conf.n_ego_vehicles)
        assert conf.n_ego_vehicles <= conf.n_vehicles_max

        sumo_scenario = cls()
        scenario_name = get_scenario_name_from_netfile(net_file)
        out_folder = os.path.join(conf.scenarios_path, scenario_name)
        pathlib.Path(out_folder).mkdir(parents=False, exist_ok=True)

        # create files
        if cr_map_path is None:
            if cr_map_converter_installed is False:
                raise EnvironmentError('This function requires the crmapconverter which is yet to be released,'
                                       'please provide a path to a CommonRoadMap using cr_map_path.')
            cr_map_path = convert_net_to_cr(net_file, out_folder)

        lanelet_network = CommonRoadFileReader(
            cr_map_path).open_lanelet_network()
        cr2sumo_converter = CR2SumoMapConverter(lanelet_network, conf)
        cr2sumo_converter._generate_routes(net_file)

        sumo_scenario.initialize(scenario_name, cr2sumo_converter.sumo_cfg_file, cr_map_path,
                                 conf.ego_start_time)
        return sumo_scenario

    def _get_net_file(self, sumo_cfg_file: str) -> str:
        """
        Gets the net file configured in the cfg file.

        :param sumo_cfg_file: SUMO config file (.sumocfg)

        :return: net-file specified in the config file
        """
        if not os.path.isfile(sumo_cfg_file):
            raise ValueError(
                "File not found: {}. Maybe scenario name is incorrect.".format(
                    sumo_cfg_file))
        tree = et.parse(sumo_cfg_file)
        file_directory = os.path.dirname(sumo_cfg_file)
        # find net-file
        all_net_files = tree.findall('*/net-file')
        if len(all_net_files) != 1:
            raise NetError(len(all_net_files))
        return os.path.join(file_directory, all_net_files[0].attrib['value'])

    def print_lanelet_net(self,
                          with_lane_id=True,
                          with_succ_pred=False,
                          with_adj=False,
                          with_speed=False) -> None:
        """
        Prints the lanelet net.

        :param with_lane_id: if true, shows the lane id.
        :param with_succ_pred: if true, shows the successors and precessors.
        :param with_adj: if true, show the adjacent lanelets.
        :param with_speed: if true, shows the speed limit of the lanelt.

        """
        from commonroad.visualization.draw_dispatch_cr import draw_object
        import matplotlib.pyplot as plt
        plt.figure(figsize=(25, 25))
        plt.gca().set_aspect('equal')
        draw_object(self.lanelet_network)
        k = len(self.lanelet_network.lanelets)
        # add annotations
        for l in self.lanelet_network.lanelets:
            # assure that text for two different lanelets starting on same position is placed differently
            # print(l.lanelet_id)
            k = k - 1
            info = ''
            if with_lane_id:
                id = 'id: ' + str(l.lanelet_id)
                plt.text(l.center_vertices[0, 0],
                         l.center_vertices[0, 1],
                         id,
                         zorder=100,
                         size=8,
                         color='r',
                         verticalalignment='top')
            if with_succ_pred:
                info = info + '\nsucc: ' + str(l.successor) + ' pred: ' + str(
                    l.predecessor)
            if with_adj:
                info = info + ' \nadj_l: ' + str(
                    l.adj_left) + '; adj_l_same_dir: ' + str(
                        l.adj_left_same_direction)
                info = info + ' \nadj_r: ' + str(
                    l.adj_right) + '; adj_r_same_dir: ' + str(
                        l.adj_right_same_direction)
            if with_speed:
                info = info + '\nspeed limit: ' + str(l.speed_limit)
            plt.plot(l.center_vertices[0, 0], l.center_vertices[0, 1], 'x')
            plt.text(l.center_vertices[0, 0],
                     l.center_vertices[0, 1],
                     info,
                     zorder=100,
                     size=8,
                     verticalalignment='top')
        plt.show()
