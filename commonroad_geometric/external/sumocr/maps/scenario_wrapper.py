from abc import ABCMeta
from commonroad.scenario.lanelet import LaneletNetwork

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2021.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class AbstractScenarioWrapper(metaclass=ABCMeta):
    # these two attributes are abstract and need to be defined by the inheriting subclass
    sumo_cfg_file = ""
    network_file = ""

    lanelet_network: LaneletNetwork = None

