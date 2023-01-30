import os
import sys

sys.path.insert(0, os.getcwd())

import argparse

from scripts.toolpaper.figures.generate_v2v_edges_plot import generate_v2v_edges_plot
from scripts.toolpaper.figures.generate_vtv_edges_plot import generate_vtv_edges_plot
from scripts.toolpaper.figures.generate_lanelet_graph_conversion_plot import generate_lanelet_graph_conversion_plot
from crgeo.common.logging import setup_logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--scenario-dir", type=str, default=INPUT_SCENARIO, help="path to scenario directory or scenario file")
    parser.add_argument("--hd", action="store_true", help="high resolution rendering")

    args = parser.parse_args()

    setup_logging()
    
    #generate_v2v_edges_plot(enable_hd=args.hd, enable_screenshots=True)
    #generate_vtv_edges_plot(enable_hd=args.hd, enable_screenshots=True)
    generate_lanelet_graph_conversion_plot()
