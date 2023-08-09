import os
import subprocess
from typing import List

from xml.dom import minidom
from xml.etree import ElementTree as et

import sumocr

if sumocr.sumo_installed:
    import sumolib

import numpy as np
import warnings

from commonroad_geometric.external.sumocr.sumo_config import EGO_ID_START
from commonroad_geometric.external.sumocr.sumo_config import DefaultConfig

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2021.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"

##
## DEPRECATED USE CONVERSION IN commonroad-map-tool
##


def find_ego_ids_by_departure_time(rou_file: str, n_ego_vehicles: int, departure_time_ego: int, ego_ids: list) -> list:
    """
    Returns ids of vehicles from route file which match desired departure time as close as possible.

    :param rou_file: path of route file
    :param n_ego_vehicles:  number of ego vehicles
    :param departure_time_ego: desired departure time ego vehicle
    :param ego_ids: if desired ids of ego_vehicle known, specify here

    """
    vehicles = sumolib.output.parse_fast(rou_file, 'vehicle', ['id', 'depart'])
    dep_times = []
    veh_ids = []
    for veh in vehicles:
        veh_ids.append(veh[0])
        dep_times.append(int(float(veh[1])))

    if n_ego_vehicles > len(veh_ids):
        warnings.warn('only {} vehicles in route file instead of {}'.format(len(veh_ids), n_ego_vehicles),stacklevel=1)
        n_ego_vehicles = len(veh_ids)

    # check if specified ids exist
    for id in ego_ids:
        if id not in veh_ids:
            warnings.warn('<generate_rou_file> id {} not in route file!'.format_map(id))
            del id

    # assign vehicles as ego by finding closest departure time
    dep_times = np.array(dep_times)
    veh_ids = np.array(veh_ids)
    greater_start_time = np.where(dep_times >= departure_time_ego)[0]
    for index in greater_start_time:
        if len(ego_ids) == n_ego_vehicles:
            break
        else:
            ego_ids.append(veh_ids[index])

    if len(ego_ids) < n_ego_vehicles:
        n_ids_missing = n_ego_vehicles - len(ego_ids)
        ego_ids.extend((veh_ids[greater_start_time[0] - n_ids_missing:greater_start_time[0]]).tolist())

    return ego_ids


def get_scenario_name_from_crfile(filepath:str) -> str:
    """
    Returns the scenario name specified in the cr file.

    :param filepath: the path of the cr file

    """
    scenario_name:str = (os.path.splitext(os.path.basename(filepath))[0]).split('.')[0]
    return scenario_name


def get_scenario_name_from_netfile(filepath:str) -> str:
    """
    Returns the scenario name specified in the net file.

    :param filepath: the path of the net file

    """
    scenario_name:str = (os.path.splitext(os.path.basename(filepath))[0]).split('.')[0]
    return scenario_name


def get_boundary_from_netfile(filepath:str) -> list:
    """
    Get the boundary of the netfile.
    :param filepath:
    :return: boundary as a list containing min_x, max_x, min_y, max_y coordinates
    """
    tree = et.parse(filepath)
    root = tree.getroot()
    location = root.find("location")
    boundary_list = location.attrib['origBoundary']  # origBoundary
    min_x, min_y, max_x, max_y = boundary_list.split(',')
    boundary = [float(min_x), float(max_x), float(min_y), float(max_y)]
    return boundary


def get_total_lane_length_from_netfile(filepath:str) -> float:
    """
    Compute the total length of all lanes in the net file.
    :param filepath:
    :return: float value of the total lane length
    """
    tree = et.parse(filepath)
    root = tree.getroot()
    total_lane_length = 0
    for lane in root.iter('lane'):
        total_lane_length += float(lane.get('length'))
    return total_lane_length


def generate_rou_file(net_file:str, out_folder:str=None, conf:DefaultConfig=DefaultConfig()) -> str:
    """
    Creates route & trips files using randomTrips generator.

    :param net_file: path of .net.xml file
    :param total_lane_length: total lane length of the network
    :param out_folder: output folder of route file (same as net_file if None)
    :param conf: configuration file for additional parameters
    :return: path of route file
    """
    if out_folder is None:
        out_folder = os.path.dirname(net_file)

    total_lane_length = get_total_lane_length_from_netfile(net_file)
    if total_lane_length is not None:
        # calculate period based on traffic frequency depending on map size
        period = 1 / (conf.max_veh_per_km * (total_lane_length / 1000) * conf.dt)
        # print('SUMO traffic generation: traffic frequency is defined based on the total lane length of the road network.')
    elif conf.veh_per_second is not None:
        # vehicles per second
        period = 1 / (conf.veh_per_second * conf.dt)
        # print('SUMO traffic generation: the total_lane_length of the road network is not available. '
        #       'Traffic frequency is defined based on equidistant depature time.')
    else:
        period = 0.5
        # print('SUMO traffic generation: neither total_lane_length nor veh_per_second is defined. '
        #       'For each second there are two vehicles generated.')
    #step_per_departure = ((conf.departure_interval_vehicles.end - conf.departure_interval_vehicles.start) / n_vehicles_max)

    # filenames
    scenario_name = get_scenario_name_from_netfile(net_file)
    rou_file = os.path.join(out_folder, scenario_name + '.rou.xml')
    trip_file = os.path.join(out_folder, scenario_name + '.trips.xml')
    add_file = os.path.join(out_folder, scenario_name + '.add.xml')

    # create route file
    cmd = ['python', os.path.join(os.path.expanduser(os.environ['SUMO_HOME']), 'tools/randomTrips.py'),
           '-n', net_file,
           '-o', trip_file,
           '-r', rou_file,
           '-b', str(conf.departure_interval_vehicles.start),
           '-e', str(conf.departure_interval_vehicles.end),
           '-p', str(period),
           '--allow-fringe',
           '--fringe-factor', str(conf.fringe_factor),
           '--trip-attributes=departLane=\"best\" departSpeed=\"max\" departPos=\"base\" '
           ]
    # if os.path.isfile(add_file):
    #     cmd.extend(['--additional-files', add_file])

    try:
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    if conf.n_ego_vehicles != 0:
        #get ego ids and add EGO_ID_START prefix
        ego_ids = find_ego_ids_by_departure_time(rou_file, conf.n_ego_vehicles, conf.departure_time_ego, conf.ego_ids)
        write_ego_ids_to_rou_file(rou_file, ego_ids)

    return rou_file


def add_params_in_rou_file(rou_file:str, driving_params:dict=DefaultConfig.driving_params) -> None:
    """
    Add parameters for the vType setting in the route file generated by SUMO. Parameters are sampled from uniform distribution.
    :param rou_file: the route file to be modified
    :param driving_params: dictionary with driving parameter as keys and interval of sampling as values
    :return:
    """
    tree = et.parse(rou_file)
    root = tree.getroot()
    vType = root.find("vType")
    if vType is not None:
        for key, value_interval in driving_params.items():
            random_value = np.random.uniform(value_interval.start, value_interval.end, 1)[0]
            vType.set(key, str("{0:.2f}".format(random_value)))
    tree.write(rou_file)


def write_ego_ids_to_rou_file(rou_file:str, ego_ids:List[int]) -> None:
    """
    Writes ids of ego vehicles to the route file.

    :param rou_file: the route file
    :param ego_ids: a list of ego vehicle ids

    """
    tree = et.parse(rou_file)
    vehicles = tree.findall('vehicle')
    ego_str = {}
    for ego_id in ego_ids:
        ego_str.update({str(ego_id): EGO_ID_START + str(ego_id)})

    for veh in vehicles:
        if veh.attrib['id'] in ego_str:
            veh.attrib['id'] = ego_str[veh.attrib['id']]

    for elem in tree.iter():
        if (elem.text):
            elem.text = elem.text.strip()
        if (elem.tail):
            elem.tail = elem.tail.strip()
    rough_string = et.tostring(tree.getroot(), encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    text = reparsed.toprettyxml(indent="\t", newl="\n")
    file = open(rou_file, "w")
    file.write(text)