import os 
import sys

# make sure $SUMO_HOME is in system pat
if 'SUMO_HOME' in os.environ:
    sumo_installed = True
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if not tools in sys.path:
        sys.path.append(tools)
else:
    sumo_installed = False

DOCKER_REGISTRY = "gitlab.lrz.de:5005/tum-cps/commonroad-sumo-interface/sumo_docker"
