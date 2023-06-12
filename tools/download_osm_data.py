import sys, os; sys.path.insert(0, os.getcwd())

import argparse
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.common.utils.seeding import get_random_seed
from commonroad_geometric.dataset.scraping import OpenStreetMapExtractor, OpenStreetMapCrawler


logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CommonRoad scenarios from OpenStreetMap data.")
    parser.add_argument("--output", default="tutorials/output/osm", type=Path,
                        help="output directory for the generated CommonRoad scenario files")
    parser.add_argument("--overwrite", action="store_true",
                        help="remove and re-create the output directory before generating scenarios")
    parser.add_argument("--logger-file", default="osm-download.logger", help="path to the logger file")
    parser.add_argument("--seed", type=int, help="integer for seeding the random number generator")
    parser.add_argument("--num-samples", type=int, default=10, help="number of CommonRoad scenarios to generate")
    parser.add_argument("--scenario-name",
                        help="map name used for the CommonRoad scenario IDs of the generated scenarios")
    parser.add_argument("--search-radius", type=float, default=10_000.0,
                        help="sample locations which are at most this distance (in meters) away from location_query")
    parser.add_argument("--scenario-size", type=float, default=200.0,
                        help="side length of the square extraction frame in meters")
    parser.add_argument("--displacement-threshold", type=float, default=500.0,
                        help="maximum distance from sampled coordinates to nearest road in meters")
    parser.add_argument("--repair-scenario", action="store_true",
                        help="repair the scenario using ")
    parser.add_argument("--noplot", dest="plot", action="store_false",
                        help="do not plot scenarios as they are generated")
    parser.add_argument("location_query", nargs="?", default="Munich",
                        help="location around which to generate scenarios")
    args = parser.parse_args()

    setup_logging(filename=args.logger_file)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = get_random_seed()
        logger.info(f"Using random seed: {seed}")

    if args.overwrite:
        shutil.rmtree(args.output, ignore_errors=True)

    callback: Optional[Callable[[Scenario], None]] = None
    if args.plot:
        def callback_fn(scenario: Scenario) -> None:
            from commonroad_geometric.plotting.plot_scenario import plot_scenario
            logger.info(f"Extracted scenario {scenario}")
            plot_scenario(scenario, show=True)
        callback = callback_fn

    if args.scenario_name is None:
        args.scenario_name = args.location_query

    crawler = OpenStreetMapCrawler(
        output_dir=args.output,
        seed=seed,
        osm_extractor=OpenStreetMapExtractor(),
        validate_scenarios=False,
    )
    async with crawler:
        location = await crawler.osm_extractor.geocode(args.location_query)

        await crawler.crawl_at_location(
            location=location,
            scenario_name=args.scenario_name,
            n=args.num_samples,
            search_radius=args.search_radius,
            scenario_size=args.scenario_size,
            displacement_threshold=args.displacement_threshold,
            repair_scenario=args.repair_scenario,
            scenario_callback=callback,
            swallow_exceptions=True,
        )


if __name__ == '__main__':
    asyncio.run(main())
