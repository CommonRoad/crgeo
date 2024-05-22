import os
import xml.etree.ElementTree as et

import warnings

from commonroad_geometric.external.sumocr.sumo_config import plot_params, ID_DICT
from typing import List, Tuple
from pathlib import Path
from commonroad_geometric.external.sumocr.sumo_config.default import SUMO_PEDESTRIAN_PREFIX, SUMO_VEHICLE_PREFIX
from commonroad.scenario.obstacle import ObstacleType, SignalState
import enum

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2021.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"

# CommonRoad obstacle type to sumo type
VEHICLE_TYPE_CR2SUMO = {
    ObstacleType.UNKNOWN: "passenger",
    ObstacleType.CAR: "passenger",
    ObstacleType.TRUCK: "truck",
    ObstacleType.BUS: "bus",
    ObstacleType.BICYCLE: "bicycle",
    ObstacleType.PEDESTRIAN: "pedestrian",
    ObstacleType.PRIORITY_VEHICLE: "vip",
    ObstacleType.PARKED_VEHICLE: "passenger",
    ObstacleType.CONSTRUCTION_ZONE: "passenger",
    ObstacleType.TRAIN: "rail",
    ObstacleType.ROAD_BOUNDARY: "custom2",
    ObstacleType.MOTORCYCLE: "motorcycle",
    ObstacleType.TAXI: "taxi",
    ObstacleType.BUILDING: "custom2",
    ObstacleType.PILLAR: "custom2",
    ObstacleType.MEDIAN_STRIP: "custom1"
}
# CommonRoad obstacle type to sumo type
VEHICLE_TYPE_SUMO2CR = {
    "DEFAULT_PEDTYPE": ObstacleType.PEDESTRIAN,
    "passenger": ObstacleType.CAR,
    "truck": ObstacleType.TRUCK,
    "bus": ObstacleType.BUS,
    "bicycle": ObstacleType.BICYCLE,
    "pedestrian": ObstacleType.PEDESTRIAN,
    "vip": ObstacleType.PRIORITY_VEHICLE,
    "rail": ObstacleType.TRAIN,
    "motorcycle": ObstacleType.MOTORCYCLE,
    "taxi": ObstacleType.TAXI,
    "custom2": ObstacleType.PILLAR,
    "custom1": ObstacleType.MEDIAN_STRIP
}


def get_route_files(config_file:Path) -> List[Path]:
    """
    Returns net-file and route-files specified in the config file.

    :param config_file: SUMO config file (.sumocfg)

    """
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    tree = et.parse(config_file)
    file_directory = config_file.parent

    # find route-files
    all_route_files = tree.findall('*/route-files')
    route_files = []
    if len(all_route_files) < 1:
        raise RouteError()
    for item in all_route_files:
        attributes = item.attrib['value'].split(',')
        for route in attributes:
            route_files.append(Path(file_directory, route))
    return route_files


def initialize_id_dicts(id_convention: dict) -> Tuple[dict, dict]:
    """
    Creates empty nested dict structure for sumo2cr and cr2sumo dicts from id_convention and returns them.

    :param id_convention: dict with mapping from object type to id start number

    """
    sumo2cr = {}
    cr2sumo = {}
    for k in id_convention:
        sumo2cr[k] = {}
        cr2sumo[k] = {}
    sumo2cr[SUMO_PEDESTRIAN_PREFIX] = {}
    sumo2cr[SUMO_VEHICLE_PREFIX] = {}
    cr2sumo[SUMO_PEDESTRIAN_PREFIX] = {}
    cr2sumo[SUMO_VEHICLE_PREFIX] = {}
    return sumo2cr, cr2sumo


def generate_cr_id(type: str, sumo_id: str, sumo_prefix: str, ids_sumo2cr: dict, max_cr_id: int) -> int:
    """
    Generates a new commonroad ID without adding it to any ID dictionary.

    :param type: one of the keys in params.id_convention; the type defines the first digit of the cr_id
    :param sumo_id: id in sumo simulation
    :param ids_sumo2cr: dictionary of ids in sumo2cr

    """
    if type not in ID_DICT:
        raise ValueError(
            '{0} is not a valid type of id_convention. Only allowed: {1}'.
            format(type, ID_DICT.keys()))
    if sumo_id in ids_sumo2cr[type]:
        # warnings.warn(
        #     'For this sumo_id there is already a commonroad id. No cr ID is generated'
        # )
        return ids_sumo2cr[type][sumo_id]
    elif sumo_id in ids_sumo2cr[sumo_prefix]:
        raise ValueError(
            'Two sumo objects of different types seem to have same sumo ID {0}. ID must be unique'
            .format(sumo_id))

    cr_id = max_cr_id + 1
    if int(str(cr_id)[0]) != ID_DICT[type]:
        cr_id = int(str(ID_DICT[type]) + "0" * len(str(cr_id)))

    return cr_id


def cr2sumo(cr_id: int, ids_cr2sumo: dict) -> int:
    """
    Takes CommonRoad ID and returns corresponding SUMO ID.

    :param ids_cr2sumo: dictionary of ids in cr2sumo

    """

    # if type(cr_id) == list:
    #     print("id: " + str(cr_id) + ": " +
    #           str(ids_cr2sumo[SUMO_PEDESTRIAN_PREFIX] + ' ' +
    #               str(ids_cr2sumo[SUMO_VEHICLE_PREFIX])))
    #     print("\n")

    if cr_id is None:
        return None
    elif cr_id in ids_cr2sumo[SUMO_VEHICLE_PREFIX]:
        return ids_cr2sumo[SUMO_VEHICLE_PREFIX][cr_id]
    elif cr_id in ids_cr2sumo[SUMO_PEDESTRIAN_PREFIX]:
        return ids_cr2sumo[SUMO_PEDESTRIAN_PREFIX][cr_id]
    return None
    # raise ValueError('Commonroad id {0} does not exist.'.format(cr_id))


def sumo2cr(sumo_id: int, ids_sumo2cr: dict) -> int:
    """
    Returns corresponding CommonRoad ID according to sumo id.

    :param sumo_id: sumo id
    :param ids_sumo2cr: dictionary of ids in sumo2cr.

    """
    if sumo_id is None:
        return None
    elif sumo_id in ids_sumo2cr[SUMO_VEHICLE_PREFIX]:
        return ids_sumo2cr[SUMO_VEHICLE_PREFIX][sumo_id]
    elif sumo_id in ids_sumo2cr[SUMO_PEDESTRIAN_PREFIX]:
        return ids_sumo2cr[SUMO_PEDESTRIAN_PREFIX][sumo_id]
    elif sumo_id == "":
        warnings.warn('Tried to convert id <empty string>. \
            Check if your net file is complete (e. g. having internal-links,...)'
                      )
    return None
    # raise ValueError('Sumo id \'%s\' does not exist.' % sumo_id)


class DummyClass:
    """Dummy class used if SUMO is not installed and traci cannot be imported"""

    def __init__(self):
        pass


class SumoSignalIndices(enum.Enum):
    """All interpretations with their respective bit indices
    ref.: https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html"""
    VEH_SIGNAL_BLINKER_RIGHT = 0
    VEH_SIGNAL_BLINKER_LEFT = 1
    VEH_SIGNAL_BLINKER_EMERGENCY = 2
    VEH_SIGNAL_BRAKELIGHT = 3
    VEH_SIGNAL_FRONTLIGHT = 4
    VEH_SIGNAL_FOGLIGHT = 5
    VEH_SIGNAL_HIGHBEAM = 6
    VEH_SIGNAL_BACKDRIVE = 7
    VEH_SIGNAL_WIPER = 8
    VEH_SIGNAL_DOOR_OPEN_LEFT = 9
    VEH_SIGNAL_DOOR_OPEN_RIGHT = 10
    VEH_SIGNAL_EMERGENCY_BLUE = 11
    VEH_SIGNAL_EMERGENCY_RED = 12
    VEH_SIGNAL_EMERGENCY_YELLOW = 13


_defined_signals = {
    # only the following signals are computed on every time step
    SumoSignalIndices.VEH_SIGNAL_BLINKER_LEFT: "indicator_left",
    SumoSignalIndices.VEH_SIGNAL_BLINKER_RIGHT: "indicator_right",
    SumoSignalIndices.VEH_SIGNAL_BRAKELIGHT: "braking_lights",
    SumoSignalIndices.VEH_SIGNAL_EMERGENCY_BLUE: "flashing_blue_lights"
}


def get_signal_state(state: int, time_step: int) -> SignalState:
    """
    Computes the CR Signal state from the sumo signals
    """
    binary = list(reversed(bin(state)[2:]))
    max_signal_index: int = max([s.value for s in SumoSignalIndices])
    bit_string: List[bool] = [binary[i] == "1" if i < len(binary) else False
                              for i in range(max_signal_index + 1)]

    args = {cr_name: bit_string[sumo_name.value]
            for sumo_name, cr_name in _defined_signals.items()}
    signal_state = SignalState(**{**args,
                                  # the following are not modelled by sumo, so have to be inserted manually
                                  **{"horn": False, "hazard_warning_lights": False},
                                  **{"time_step": time_step}})
    return signal_state


class NetError(Exception):
    """
    Exception raised if there is no net-file or multiple net-files.

    """

    def __init__(self, len):
        self.len = len

    def __str__(self):
        if self.len == 0:
            return repr('There is no net-file.')
        else:
            return repr('There are more than one net-files.')


class RouteError(Exception):
    """
    Exception raised if there is no route-file.

    """

    def __str__(self):
        return repr('There is no route-file.')


class EgoCollisionError(Exception):
    """
    Exception raised if the ego vehicle collides with another vehicle

    """

    def __init__(self, time_step=None):
        super().__init__()
        self.time_step = time_step

    def __str__(self):
        if self.time_step is not None:
            return repr(f'Ego vehicle collides at current simulation step = {self.time_step}!')
        else:
            return repr(f'Ego vehicle collides at current simulation step!')
