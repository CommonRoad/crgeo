import os
import sys

sumo_installed = False

# make sure $SUMO_HOME is in system pat
if 'SUMO_HOME' in os.environ:
    sumo_installed = True
else:
    sumo_home = os.path.join(sys.base_prefix, 'lib', f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "sumo")
    if os.path.isdir(sumo_home):
        os.environ['SUMO_HOME'] = sumo_home
        sumo_installed = True

if sumo_installed:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if not tools in sys.path:
        sys.path.append(tools)

