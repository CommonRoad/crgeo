"""
Configuring paths for the sumo binaries
"""
import os

import crgeo.external.sumocr as sumocr
# set installation locations
SUMO_GUI_BINARY = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui') if sumocr.sumo_installed else None

# path to binary of adapted sumo repository (see readme)
SUMO_BINARY = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo') if sumocr.sumo_installed else None

# by default port 8873 is used, you can modify the port number
TRACI_PORT = 8873

# default sumo configuration
DEFAULT_CFG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '', '.sumo.cfg'))
