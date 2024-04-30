import asyncio
import logging
import math
import pickle
import random
import urllib.error
import uuid
import warnings
import xml
from asyncio import Task
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import geopy.distance
import iso3166
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.scenario.scenario import Location, Scenario, ScenarioID, Tag
from geopy import Point
from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeopyError
from geopy.geocoders import Nominatim

import commonroad_geometric.external.map_conversion.osm2cr.config as OSM_CONFIG
import commonroad_geometric.external.map_conversion.osm2cr.converter_modules.converter as converter
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.io_extensions.lanelet_network import remove_empty_intersections
from commonroad_geometric.common.utils.filesystem import slugify
from commonroad_geometric.dataset.scraping.rate_limiter import AsyncRateLimiter
from commonroad_geometric.external.map_conversion.osm2cr.converter_modules.cr_operations.cleanup import sanitize
from commonroad_geometric.external.map_conversion.osm2cr.converter_modules.cr_operations.export import create_scenario_intermediate
from commonroad_geometric.external.map_conversion.osm2cr.converter_modules.intermediate_operations.intermediate_format import IntermediateFormat
from commonroad_geometric.external.map_conversion.osm2cr.converter_modules.osm_operations import downloader

log = logging.getLogger(__name__)


class MapDownloadFailed(Exception):
    pass


class OpenStreetMapExtractor:

    DEFAULT_MAP_DOWNLOAD_URLS = [
        # https://wiki.openstreetmap.org/wiki/API_v0.6#Retrieving_map_data_by_bounding_box:_GET_.2Fapi.2F0.6.2Fmap
        "https://api.openstreetmap.org/api/0.6/map?bbox={lon1},{lat1},{lon2},{lat2}",

        # https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
        "https://lz4.overpass-api.de/api/map?bbox={lon1},{lat1},{lon2},{lat2}",
        "https://z.overpass-api.de/api/map?bbox={lon1},{lat1},{lon2},{lat2}",
        "https://overpass.kumi.systems/api/map?bbox={lon1},{lat1},{lon2},{lat2}",
    ]

    def __init__(
        self,
        geolocator_min_delay: float = 1.0,
        geolocator_max_retries: int = 4,
        geolocator_retry_delay: float = 20.0,
        map_download_urls: Optional[List[str]] = None,
        map_download_min_delay: float = 1.0,
        map_download_max_retries: int = 2,
        map_download_retry_delay: float = 20.0,
    ):
        def rate_limiter_func(func: Callable, *args, **kwargs):
            return func(*args, **kwargs)

        self._geolocator = Nominatim(
            user_agent=self.__class__.__name__,
            adapter_factory=AioHTTPAdapter,
        )
        self._geolocator_rate_limiter = AsyncRateLimiter(
            func=rate_limiter_func,
            min_delay_seconds=geolocator_min_delay,
            max_retries=geolocator_max_retries,
            exception_wait_seconds=geolocator_retry_delay,
            retry_exceptions=(GeopyError,),
            swallow_exceptions=False,
        )

        def map_download_exception_callback(e: Exception):
            # check for 509 Bandwidth Limit Exceeded server error
            return isinstance(e, urllib.error.HTTPError) and e.code != 509

        self._download_counter = 0
        if map_download_urls is None:
            map_download_urls = self.DEFAULT_MAP_DOWNLOAD_URLS
        self._map_download_urls = map_download_urls
        self._map_download_rate_limiters = [
            AsyncRateLimiter(
                func=rate_limiter_func,
                min_delay_seconds=map_download_min_delay,
                max_retries=map_download_max_retries,
                exception_wait_seconds=map_download_retry_delay,
                retry_exceptions=(asyncio.TimeoutError, urllib.error.HTTPError),
                exception_callback=map_download_exception_callback,
                swallow_exceptions=True,
            )
            for _ in range(len(map_download_urls))
        ]

    async def geocode(self, query: str) -> Optional[geopy.Location]:
        """Address to coordinates."""
        return await self._geolocator_rate_limiter(self._geolocator.geocode, query)

    async def reverse_geocode(self, query: Union[Point, str]) -> Optional[geopy.Location]:
        """Coordinates to address."""
        return await self._geolocator_rate_limiter(
            self._geolocator.reverse,
            query,
            language="en",
            zoom=17,
        )

    async def extract_from_latlon(
        self,
        lat: float,
        lon: float,
        scenario_size: float,
        scenario_id: Optional[ScenarioID] = None,
        repair_scenario: bool = False,
        remove_traffic_lights_and_signs: bool = False,
    ) -> Tuple[Scenario, IntermediateFormat]:
        """Downloads scenario using latitude and longitude as origin."""
        log.info(
            "Extracting scenario %s at lat=%f, lon=%f with scenario size=%f",
            scenario_id, lat, lon, scenario_size,
        )

        if scenario_id is None:
            pos = await self.reverse_geocode(Point(latitude=lat, longitude=lon))
            if pos is None:
                country_id = "ZAM"
            else:
                country_id = iso3166.countries_by_alpha2[pos.raw["address"]["country_code"].upper()].alpha3
            scenario_id = ScenarioID(
                country_id=country_id,
                # map_name=f"lat{lat:.4f}-lon{lon:.4f}-radius{radius:.0f}m",
                map_name="latlonextract_" + str(uuid.uuid4()),
                map_id=0,
            )

        lon1, lat1, lon2, lat2 = downloader.get_frame(
            lat=lat,
            lon=lon,
            radius=scenario_size / 2,
        )

        with NamedTemporaryFile() as f:
            success = await self._download_map(f, lon1, lat1, lon2, lat2)  # type: ignore
            if not success:
                raise MapDownloadFailed(f"Failed to download the map for scenario %s", scenario_id)

            graph_scenario = converter.GraphScenario(f.name)

        # TODO this is extremely hacky
        with warnings.catch_warnings(record=True) as ws:
            scenario, intermediate = create_scenario_intermediate(graph_scenario.graph)
        scenario_invalid = any(
            all((isinstance(warning.message, UserWarning),
                 str(warning.message).startswith("Lanelet "),
                 str(warning.message).endswith(" invalid")))
            for warning in ws
        )
        if scenario_invalid:
            raise ValueError("Scenario contains invalid lanelet")

        if repair_scenario:
            from crdesigner.map_validation_repairing.map_validator_repairer import MapValidatorRepairer
            repairer = MapValidatorRepairer(scenario)
            repairer.validate_and_repair(iterations=100)

        sanitize(scenario)
        if remove_traffic_lights_and_signs:
            for x in scenario.lanelet_network.traffic_signs:
                try:
                    scenario.remove_traffic_sign(x)
                except KeyError:
                    pass
            for x in scenario.lanelet_network.traffic_lights:
                try:
                    scenario.remove_traffic_light(x)
                except KeyError:
                    pass

        scenario.scenario_id = scenario_id
        scenario.location = Location(gps_latitude=lat, gps_longitude=lon)

        remove_empty_intersections(scenario.lanelet_network)
        scenario.lanelet_network.cleanup_lanelet_references()
        # if self._validate_scenarios:
        #     create_graph_from_lanelet_network(
        #         scenario.lanelet_network,
        #         validate=True
        #     )

        return scenario, intermediate

    async def extract_from_address(
        self,
        address: str,
        scenario_size: float,
        repair_scenario: bool = False,
        remove_traffic_lights_and_signs: bool = False,
    ) -> Tuple[Scenario, IntermediateFormat]:
        """Downloads scenario using specified address as origin.

        Args:
            address (str): Human-readable address.
        """
        location = await self.geocode(address)
        if location is None:
            raise LookupError(f"Could not geocode '{address}'")

        scenario_id = ScenarioID(
            country_id="ZAM",
            map_name=slugify(address, replacement="", title=True),
            map_id=0
        )
        return await self.extract_from_latlon(
            lat=location.latitude,
            lon=location.longitude,
            scenario_id=scenario_id,
            scenario_size=scenario_size,
            repair_scenario=repair_scenario,
            remove_traffic_lights_and_signs=remove_traffic_lights_and_signs,
        )

    async def _download_map(self, f: BinaryIO, lon1: float, lat1: float, lon2: float, lat2: float) -> bool:
        async def download():
            async with session.get(url) as response:
                return await response.read()

        session = aiohttp.ClientSession(
            headers={
                "User-Agent": self.__class__.__name__,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._download_counter += 1
        url_index = self._download_counter % len(self._map_download_urls)
        url_template = self._map_download_urls[url_index]
        url = url_template.format(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
        async with session:
            data = await self._map_download_rate_limiters[url_index](download)

        if data is None:
            return False

        # copied from downloader.write_bounds_to_file
        root = xml.etree.ElementTree.fromstring(data)
        tag = xml.etree.ElementTree.SubElement(root, "custom_bounds")
        tag.attrib["lon1"] = str(lon1)
        tag.attrib["lon2"] = str(lon2)
        tag.attrib["lat1"] = str(lat1)
        tag.attrib["lat2"] = str(lat2)
        tree = xml.etree.ElementTree.ElementTree(element=root)
        tree.write(f, encoding="utf-8", xml_declaration=True)
        return True

    async def close(self) -> None:
        await self.__aexit__(None, None, None)

    async def __aenter__(self) -> "OpenStreetMapExtractor":
        await self._geolocator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._geolocator.__aexit__(exc_type, exc_val, exc_tb)


class OpenStreetMapCrawler(AutoReprMixin):
    """Class that enables scraping of random scenarios within a geographical area.
    Builds upon the scenario conversion functionality from
    https://commonroad-scenario-designer.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        output_dir: Path,
        seed: int,
        osm_extractor: OpenStreetMapExtractor,
        validate_scenarios: bool = True,
    ) -> None:
        output_dir.mkdir(exist_ok=True, parents=True)
        self._output_dir = output_dir

        self._state_file = output_dir / "osm_crawler_state.pickle"
        self._crawled_scenarios: Dict[int, dict] = {}

        self._initial_seed = seed
        self._seed = random.Random(self._initial_seed).getrandbits(32)
        self._validate_scenarios = validate_scenarios
        self._osm_extractor = osm_extractor

        self._load_state()

        log.debug("Initialized OpenStreetMapCrawler with output_dir %s", output_dir)

    @property
    def osm_extractor(self):
        return self._osm_extractor

    def _save_state(self) -> None:
        state = {
            "scenarios": self._crawled_scenarios,
        }
        with self._state_file.open("wb") as f:
            pickle.dump(state, f)

    def _load_state(self) -> None:
        try:
            with self._state_file.open("rb") as f:
                state = pickle.load(f)
        except FileNotFoundError:
            return

        self._crawled_scenarios = state["scenarios"]

    async def crawl_at_location(
        self,
        location: geopy.Location,
        scenario_name: str,
        n: int = -1,
        search_radius: float = 1000.0,
        scenario_size: float = 100.0,
        displacement_threshold: float = 100.0,
        scenario_filter: Optional[Callable[[Scenario], bool]] = None,
        scenario_callback: Optional[Callable[[Scenario], None]] = None,
        repair_scenario: bool = False,
        remove_traffic_lights_and_signs: bool = False,
        swallow_exceptions: bool = True,
        max_concurrent_tasks: int = 10,
    ) -> None:
        """Randomly downloads OSM scenarios from the specified city
        and saves the converted CommonRoad scenario files.

        Args:
            location (Location):
                Location around which to collect scenarios.
            scenario_name (Path):
                Name of the CommonRoad scenario, e.g. Munich.
            n (int, optional):
                Number of scenarios to save. Defaults to -1.
            search_radius (float, optional):
                Search radius in meters. Defaults to 1000.0.
            scenario_size (float, optional):
                Side length of the square extraction frame in meters. Defaults to 100.0.
            displacement_threshold (float, optional):
                Maximum distance from sampled coordinates to nearest road in meters. Defaults to 100.
            scenario_filter (Callable[[Scenario], bool], optional):
                Optional boolean filter for downloaded scenarios. Defaults to None.
            scenario_callback (Callable[[Scenario], None], optional):
                Optional callback for downloaded scenarios. Defaults to None.
            repair_scenario (bool, optional):
                Repair scenario using MapValidatorRepairer from the commonroad-scenario-designer package
            remove_traffic_lights_and_signs (bool, optional):
                Remove traffic lights and signs (can cause failures if included). Defaults to False.
            swallow_exceptions (bool, optional):
                Ignore exceptions originating from downloading and conversion. Defaults to True.
            max_concurrent_tasks (int, optional):
                Maximum number of concurrent crawling tasks. Defaults to 10.
        """
        log.info("Crawling at location %s for scenario %s with search radius %f and n=%d",
                 location, scenario_name, search_radius, n)
        collect_infinite = n < 0

        map_name = slugify(scenario_name, replacement="", title=True)

        count = 0
        success_count = 0
        crawl_tasks: Set[Task[bool]] = set()
        while collect_infinite or success_count < n:
            if count in self._crawled_scenarios:
                count += 1
                continue

            if len(crawl_tasks) == max_concurrent_tasks or not collect_infinite and success_count + len(crawl_tasks) == n:
                done_tasks, crawl_tasks = await asyncio.wait(crawl_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    try:
                        crawl_success = task.result()
                        if crawl_success:
                            success_count += 1
                    except Exception:
                        if swallow_exceptions:
                            log.exception("Crawling task raised an exception")
                        else:
                            raise

                continue

            task = asyncio.create_task(
                self._crawl(
                    map_name, location, count, search_radius, scenario_size, displacement_threshold,
                    scenario_filter, scenario_callback, repair_scenario, remove_traffic_lights_and_signs,
                    swallow_exceptions,
                )
            )
            crawl_tasks.add(task)
            count += 1

        if crawl_tasks:
            await asyncio.wait(crawl_tasks, return_when=asyncio.ALL_COMPLETED)
            for task in crawl_tasks:
                try:
                    task.result()
                except Exception:
                    if swallow_exceptions:
                        log.exception("Crawling task raised an exception")
                    else:
                        raise

    async def _crawl(
        self,
        map_name: str,
        location: geopy.Location,
        count: int,
        search_radius: float,
        scenario_size: float,
        displacement_threshold: float,
        scenario_filter: Optional[Callable[[Scenario], bool]],
        scenario_callback: Optional[Callable[[Scenario], None]],
        repair_scenario: bool,
        remove_traffic_lights_and_signs: bool,
        swallow_exceptions: bool,
    ) -> bool:
        rng = random.Random(self._seed + count)
        bearing = rng.random() * 360
        distance = math.sqrt(rng.random()) * search_radius

        geopy_distance = geopy.distance.distance(meters=distance)
        start_pos = geopy_distance.destination(point=location.point, bearing=bearing)

        # find the nearest address
        pos = await self._osm_extractor.reverse_geocode(start_pos)
        if pos is None:
            return False  # reverse lookup failed

        displacement = geopy.distance.distance(start_pos, pos.point).meters
        if displacement > displacement_threshold:
            return False  # too far from sampled location

        scenario_id = ScenarioID(
            country_id=iso3166.countries_by_alpha2[pos.raw["address"]["country_code"].upper()].alpha3,
            map_name=map_name,
            map_id=count,
        )
        try:
            scenario, intermediate = await self._osm_extractor.extract_from_latlon(
                lat=pos.latitude,
                lon=pos.longitude,
                scenario_size=scenario_size,
                scenario_id=scenario_id,
                repair_scenario=repair_scenario,
                remove_traffic_lights_and_signs=remove_traffic_lights_and_signs,
            )

            if scenario_filter is not None and not scenario_filter(scenario):
                raise ValueError("Scenario not valid according to filter")

            if scenario_callback is not None:
                scenario_callback(scenario)

            self._write_scenario(scenario, intermediate)
            self._crawled_scenarios[count] = dict(scenario_id=scenario_id)
            log.info(
                "Successfully created scenario %s at %s (bearing %f deg, distance %f m)",
                scenario_id, pos, bearing, distance,
            )
            return True

        except Exception as e:
            log.exception("Failed to extract scenario %d", count)
            if not swallow_exceptions:
                raise e

            self._crawled_scenarios[count] = dict(error=str(e))
            return False

        finally:
            self._save_state()

    def _write_scenario(self, scenario: Scenario, intermediate: IntermediateFormat) -> None:
        file_writer = CommonRoadFileWriter(
            scenario=scenario,
            planning_problem_set=intermediate.get_dummy_planning_problem_set(),
            author=self.__class__.__name__,
            affiliation=OSM_CONFIG.AFFILIATION,
            source=OSM_CONFIG.SOURCE,
            tags={Tag.URBAN},
            location=scenario.location,
        )

        # write scenario to file with planning problem
        output_path = Path(self._output_dir,f"{scenario.scenario_id}.xml")
        file_writer.write_to_file(
            str(output_path),
            overwrite_existing_file=OverwriteExistingFile.ALWAYS,
            check_validity=self._validate_scenarios,
        )

        if self._validate_scenarios:
            # test file load to see if it raises an exception
            CommonRoadFileReader(filename=str(output_path)).open()

    async def close(self) -> None:
        await self.__aexit__(None, None, None)

    async def __aenter__(self) -> "OpenStreetMapCrawler":
        await self._osm_extractor.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._osm_extractor.__aexit__(exc_type, exc_val, exc_tb)
