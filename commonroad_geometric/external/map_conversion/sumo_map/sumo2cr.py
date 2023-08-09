import os
import subprocess

from commonroad.scenario.scenario import Scenario
from commonroad_geometric.external.map_conversion.opendrive.opendrive_conversion.network import Network
from commonroad_geometric.external.map_conversion.opendrive.opendrive_parser.parser import parse_opendrive
from lxml import etree


def convert_net_to_cr(net_file: str, verbose: bool = False) -> Scenario:
    """
    Converts .net file to CommonRoad xml using netconvert and OpenDRIVE 2 Lanelet Converter.

    :param net_file: path of .net.xml file
    :param verbose: Boolean indicating whether status should be printed to console

    :return: commonroad map file
    """
    assert isinstance(net_file, str)

    out_folder_tmp = os.path.dirname(net_file)

    # filenames
    scenario_name = _get_scenario_name_from_netfile(net_file)
    opendrive_file = os.path.join(out_folder_tmp, scenario_name + '.xodr')

    # convert to OpenDRIVE file using netconvert
    subprocess.check_output(['netconvert', '-s', net_file,
                             '--opendrive-output', opendrive_file,
                             '--junctions.scurve-stretch', '1.0'])
    if verbose:
        print('converted to OpenDrive (.xodr)')

    # convert to commonroad using opendrive2lanelet
    # import, parse and convert OpenDRIVE file
    with open(opendrive_file, "r") as fi:
        open_drive = parse_opendrive(etree.parse(fi).getroot())

    road_network = Network()
    road_network.load_opendrive(open_drive)
    scenario = road_network.export_commonroad_scenario()
    if verbose:
        print('converted to Commonroad (.cr.xml)')

    return scenario


def _get_scenario_name_from_netfile(filepath: str) -> str:
    """
    Returns the scenario name specified in the net file.

    :param filepath: the path of the net file

    """
    scenario_name: str = (os.path.splitext(os.path.basename(filepath))[0]).split('.')[0]
    return scenario_name
