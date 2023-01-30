import copy
import logging
import math
import random
import sys
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Optional, Union, TYPE_CHECKING

import numpy as np
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario, ScenarioID
from commonroad.scenario.trajectory import State, Trajectory
from numpy.random import RandomState, RandomState

from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.common.io_extensions.lanelet_network import find_lanelet_by_id
from crgeo.external.sumocr.interface.ego_vehicle import EgoVehicle
from crgeo.external.sumocr.interface.util import *
from crgeo.external.sumocr.maps.scenario_wrapper import AbstractScenarioWrapper
from crgeo.external.sumocr.sumo_config import DefaultConfig, EGO_ID_START
from crgeo.external.sumocr.sumo_config.pathConfig import SUMO_GUI_BINARY
from crgeo.simulation.interfaces.interactive.sumo_commandline_config import SumoCommandLineConfig

logger = logging.getLogger(__name__)

# make sure $SUMO_HOME is in system pat
if 'SUMO_HOME' in os.environ:
    sumo_installed = True
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if not tools in sys.path:
        sys.path.append(tools)
else:
    sumo_installed = False


with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def sample(val: Union[Interval, float]) -> float:
    if isinstance(val, Interval):
        assert 0 <= val.start <= val.end, f"All values in the interval need to be positive: {val}"
        return float(np.random.uniform(val.start, val.end))
    else:
        return val


class _CRSumoSimulation(AutoReprMixin):
    """
    Class for interfacing between the SUMO simulation and CommonRoad.

    Modified version of https://gitlab.lrz.de/tum-cps/commonroad-sumo-interface
    supporting continuous simulation.
    """

    def __init__(self, silent: bool = True, include_lanes: Set[int] = None,):

        try:
            import traci
            self._traci = traci
            from traci._person import PersonDomain
            from traci._vehicle import VehicleDomain
            from traci._vehicletype import VehicleTypeDomain
            from traci._simulation import SimulationDomain
            from traci._edge import EdgeDomain
            from traci._route import RouteDomain
            from traci._lane import LaneDomain
            from traci.exceptions import TraCIException, FatalTraCIError
        except ModuleNotFoundError as exp:
            PersonDomain = DummyClass
            VehicleDomain = DummyClass
            VehicleTypeDomain = DummyClass
            EdgeDomain = DummyClass
            SimulationDomain = DummyClass
            RouteDomain = DummyClass
            LaneDomain = DummyClass

        from crgeo.external.sumocr.converter import CR2SumoMapConverter
        self._map_converter: Optional[CR2SumoMapConverter] = None

        """Init empty object"""
        self.silent = silent
        self.include_lanes = include_lanes
        self.dt = None
        self._running = False
        self.dt_sumo = None
        self.delta_steps = None
        # {time_step: {obstacle_id: state}}
        self.simulationdomain: SimulationDomain = SimulationDomain()
        self.vehicledomain: VehicleDomain = VehicleDomain()
        self.vehicletypedomain: VehicleTypeDomain = VehicleTypeDomain()
        self.persondomain: PersonDomain = PersonDomain()
        self.routedomain: RouteDomain = RouteDomain()
        self.edgedomain: EdgeDomain = EdgeDomain()
        self.lanedomain: LaneDomain = LaneDomain()
        self.reset_variables()

    def reset_variables(self):
        self.planning_problem_set: PlanningProblemSet = None
        self.obstacle_states: Dict[int, Dict[int, State]] = defaultdict(lambda: dict())
        self.obstacle_trajectories: Dict[int, List[State]] = defaultdict(list)
        self._current_time_step = 0
        self._map_converter = None
        self.ids_sumo2cr, self.ids_cr2sumo = initialize_id_dicts(ID_DICT)
        self._max_cr_id = 0 # keep track of all IDs in CR scenario

        # veh_id -> List[SignalState]
        self.signal_states: Dict[int, List[SignalState]] = defaultdict(list)
        self.obstacle_shapes: Dict[int, Rectangle] = dict()
        self.obstacle_types: Dict[int, ObstacleType] = dict()
        self.obstacle_lane_ids: Dict[int, int] = dict()
        self.cached_position = {}  # caches position for orientation computation
        self._scenarios: AbstractScenarioWrapper = None
        self.ego_vehicles: Dict[int, EgoVehicle] = dict()
        self.conf = DefaultConfig()
        self._silent = False
        # enables dummy synchronization of ego vehicles without planner for testing
        self.dummy_ego_simulation = False
        # ego sync parameters
        self._lc_duration_max = 10
        self._lc_counter = 0  # Count how many steps are counter for the lane change
        self._lc_inaction = 0  # Flag indicates that SUMO is performing a lane change for the ego
        self.lateral_position_buffer = dict()  # stores lateral position [ego_vehicle_id,float]
        self._traci_label = None
        self.initialized = False

    @property
    def scenarios(self):
        return self._scenarios

    @property
    def ego_vehicle_first(self) -> EgoVehicle:
        return next(iter(self.ego_vehicles.values()))

    @scenarios.setter
    def scenarios(self, scenarios: AbstractScenarioWrapper):
        def max_lanelet_network_id(lanelet_network: LaneletNetwork) -> int:
            max_lanelet = np.max([l.lanelet_id for l in lanelet_network.lanelets]) \
                if lanelet_network.lanelets else 0
            max_intersection = np.max([i.intersection_id for i in lanelet_network.intersections]) \
                if lanelet_network.intersections else 0
            max_traffic_light = np.max([t.traffic_light_id for t in lanelet_network.traffic_lights]) \
                if lanelet_network.traffic_lights else 0
            max_traffic_sign = np.max([t.traffic_sign_id for t in lanelet_network.traffic_signs]) \
                if lanelet_network.traffic_signs else 0
            return np.max([max_lanelet, max_intersection, max_traffic_light, max_traffic_sign]).item()

        if self.planning_problem_set is not None:
            max_pp= max(list(self.planning_problem_set.planning_problem_dict.keys()))
        else:
            max_pp = 0

        self._max_cr_id = max(max_pp,
                              max_lanelet_network_id(scenarios.lanelet_network))
        self._scenarios = scenarios

    def initialize(self, conf: DefaultConfig,
                   scenario_wrapper: AbstractScenarioWrapper,
                   planning_problem_set: PlanningProblemSet = None,
                   presimulation_steps: int = 10) -> None:
        """
        Reads scenario files, starts traci simulation, initializes vehicles, conducts pre-simulation.

        :param conf: configuration object. If None, use default configuration.
        :param scenario_wrapper: handles all files required for simulation. If None it is initialized with files
            folder conf.scenarios_path + conf.scenario_name
        :param planning_problem_set: initialize initial state of ego vehicles

        """
        if conf is not None:
            self.conf = conf

        # assert isinstance(scenario_wrapper, AbstractScenarioWrapper), \
        #     f'scenario_wrapper expected type AbstractScenarioWrapper or None, but got type {type(scenario_wrapper)}'
        self.scenarios = scenario_wrapper
        self.dt = self.conf.dt
        self.dt_sumo = self.conf.dt / self.conf.delta_steps
        self.delta_steps = self.conf.delta_steps
        self.planning_problem_set = planning_problem_set

        if self._map_converter is None:
            from crgeo.external.sumocr.converter import CR2SumoMapConverter
            self._map_converter = CR2SumoMapConverter(self.scenarios, conf)
            self._map_converter.lanelet_network = self.scenarios.lanelet_network
            self._map_converter._convert_map()
            veh_distr_sum = sum(self.conf.veh_distribution.values())
            self._veh_distr = {k: v/veh_distr_sum for k, v in self.conf.veh_distribution.items()}

        import tempfile
        import shutil

        dirpath = tempfile.mkdtemp()

        try:
            self._map_converter.create_sumo_files(dirpath)

            if self.conf.with_sumo_gui:
                logger.warning(
                    'Continuous lane change currently not implemented for sumo-gui.'
                )
                cmd = [
                    SUMO_GUI_BINARY, "-c", self.scenarios.sumo_cfg_file,
                    "--step-length",
                    str(self.dt_sumo), "--ignore-route-errors",
                    "--lateral-resolution",
                    str(self.conf.lateral_resolution),
                ]
                raise NotImplementedError()

            else:
                cmd_config = SumoCommandLineConfig(
                    step_length=self.dt_sumo,
                    lateral_resolution=self.conf.lateral_resolution,
                    time_to_impatience=1,
                    waiting_time_memory=1,
                    ignore_junction_blocker=1,
                    tls__all_off=True
                )
                if hasattr(self.scenarios, 'sumo_cfg_file'):
                    print(f"Starting SUMO simulation from {self.scenarios.sumo_cfg_file}")
                    cmd_config.configuration_file = self.scenarios.sumo_cfg_file
                else:
                    cmd_config.net_file = f'{dirpath}/{self.conf.scenario_name}.net.xml'
                if self.silent:
                    cmd_config.verbose = False
                    cmd_config.no_warnings = True

            if self.conf.lateral_resolution > 0.0:
                cmd_config.lanechange__duration = 0

            if self.conf.random_seed:
                np.random.seed(self.conf.random_seed)
                cmd_config.seed = self.conf.random_seed

            try:
                cmd_config.save_config()
                cmd = cmd_config.generate_command()
                self._traci.start(cmd, label=self.traci_label)
            except Exception:
                logger.warn(f"self._traci.start failing after cmd={cmd}")
                raise
            self._running = True
            # simulate until ego_time_start
            self.presimulation_silent(presimulation_steps)
        finally:
            shutil.rmtree(dirpath)

        self.initialized = True

        edge_ids = [edge_id for edge_id in self.edgedomain.getIDList() if edge_id[0] != ':']
        self.entrance_edges = []
        self.exit_edges = []
        for edge_id in edge_ids:

            is_exit_edge = True
            is_entrance_edge = True
            for other_edge_id in edge_ids:
                if other_edge_id == edge_id:
                    continue
                route_from = self.simulationdomain.findRoute(edge_id, other_edge_id)
                route_to = self.simulationdomain.findRoute(other_edge_id, edge_id)
                if route_from.length > 0.0:
                    is_exit_edge = False
                if route_to.length > 0.0:
                    is_entrance_edge = False

            if is_exit_edge:
                self.exit_edges.append(edge_id)
            elif is_entrance_edge:
                self.entrance_edges.append(edge_id)

        self.routes: List[Tuple[str, List[str]]] = []
        for entrance_edge in self.entrance_edges:
            for exit_edge in self.exit_edges:
                route = self.simulationdomain.findRoute(entrance_edge, exit_edge)
                if route.edges:
                    route_id = '-'.join(route.edges)
                    self.routedomain.add(route_id, route.edges)
                    self.routes.append((route_id, list(route.edges)))

        # removing original vehicles
        for vehicle_id in self.vehicledomain.getIDList():
            self.vehicledomain.remove(vehicle_id)
        # if routes is not None:
        #     for i, route in enumerate(routes):
        #         try:
        #             self.routedomain.add('!' + str(i), list(map(str, route)))
        #         except TraCIException:
        #             pass

        #self.init_ego_vehicle()
        # initializes vehicle positions (and ego vehicles, if defined in .rou file)
        # if self.planning_problem_set is not None:
        #     if len(self.ego_vehicles) > 0:
        #         logger.warning('<_CRSumoSimulation/init_ego_vehicles> Ego vehicles are already defined through .rou'
        #                             'file and planning problem!')
        #     self.init_ego_vehicles_from_planning_problem(self.planning_problem_set)
        # else:
        #     self.dummy_ego_simulation = True

    def _init_logging(self):
        # Create a custom logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level=getattr(logging, self.conf.logging_level))
        if not logger.hasHandlers():
            # Create handlers
            c_handler = logging.StreamHandler()

            # Create formatters and add it to handlers
            c_format = logging.Formatter('<%(name)s.%(funcName)s> %(message)s')
            c_handler.setFormatter(c_format)

            # Add handlers to the logger
            logger.addHandler(c_handler)

        return logger

    def init_ego_vehicle(self, planning_problem: Optional[PlanningProblem] = None) -> None:
        """
        Initializes the ego vehicles according to planning problem set.

        :param planning_problem_set: The planning problem set which defines the ego vehicles.
        """
        assert self.initialized, 'Initialize the _CRSumoSimulation first!'

        self.dummy_ego_simulation = False


        width = self.conf.ego_veh_width
        length = self.conf.ego_veh_length

        new_id = self._create_sumo_id([])
        sumo_id = EGO_ID_START + str(new_id)
        cr_id = self._create_cr_id(type='egoVehicle', sumo_id=sumo_id, sumo_prefix=EGO_ID_START)

        self._add_vehicle_to_sim(sumo_id)
        self._traci.simulationStep()

        self.ids_sumo2cr[EGO_ID_START][sumo_id] = cr_id
        self.ids_sumo2cr['vehicle'][sumo_id] = cr_id
        self.ids_cr2sumo[EGO_ID_START][cr_id] = sumo_id
        self.ids_cr2sumo['vehicle'][cr_id] = sumo_id
        initial_state = copy.deepcopy(planning_problem.initial_state)
        initial_state.time_step = self.current_time_step
        self._add_ego_vehicle(EgoVehicle(cr_id, initial_state,
                                            self.conf.delta_steps, width, length,
                                            planning_problem)
        )

    def _add_vehicle_to_sim(self, sumo_id: str) -> None:
        # retrieve arbitrary route id for initialization (will not be used by interface)
        generic_route_id = self.routedomain.getIDList()[0]
        self.vehicledomain.add(sumo_id, generic_route_id, typeID="DEFAULT_VEHTYPE",
            depart=None,
            departLane='first', departPos="base",
            departSpeed=0.0,
            arrivalLane="current", arrivalPos="max",
            arrivalSpeed="current",
            fromTaz="", toTaz="", line="", personCapacity=0,
            personNumber=0
        )

    def spawn_vehicle(
        self,
        init_speed: float = 0.0,
        max_speed: float = None,
        p_wants_lane_change: float = 0.0,
        depart_pos: float = None,
        lanelet_id: int = None,
        rng: RandomState = None,
        obstacle_type: ObstacleType = None,
    ) -> Optional[str]:
        from traci.exceptions import TraCIException
        rng = rng if rng is not None else np.random

        new_id = rng.randint(100000000, 999999999)
        sumo_id = SUMO_VEHICLE_PREFIX + str(new_id)
        if lanelet_id is None:
            route_id, route_edges = rng.choice(self.routes)
            n_depart_lanes = self.edgedomain.getLaneNumber(route_edges[0])
            if n_depart_lanes == 1:
                depart_lane = 0
            else:
                depart_lane = rng.randint(0, n_depart_lanes-1)
        else:
            route_id_candidates = [
                (route_id, route_edges)
                for route_id, route_edges in self.routes
                if int(route_edges[0]) == self._map_converter.lanelet_id2edge_id[lanelet_id]
            ]
            if not route_id_candidates:
                logger.error("Unable to spawn vehicle: no suitable routes in SUMO simulation")
                return None

            route_id, route_edges = random.choice(route_id_candidates)
            depart_lane = self._map_converter.lanelet_id2edge_lane_id[lanelet_id]

        if rng.random() <= p_wants_lane_change:
            n_exit_lanes = self.edgedomain.getLaneNumber(route_edges[-1])
            if n_exit_lanes == 1:
                exit_lane = 0
            else:
                exit_lane = rng.randint(0, n_exit_lanes-1)
            arrival_lane = str(exit_lane)
        else:
            arrival_lane = "current"

        if obstacle_type is None:
            obstacle_type = rng.choice(list(self._veh_distr.keys()), p=list(self._veh_distr.values()))
        vehicle_class = VEHICLE_TYPE_CR2SUMO[obstacle_type]

        try:
            self.vehicledomain.add(
                sumo_id,
                route_id,
                typeID='DEFAULT_VEHTYPE',
                depart=None,
                departLane=str(depart_lane),
                departPos=depart_pos if depart_pos is not None else "base",
                departSpeed=str(init_speed),
                arrivalLane=arrival_lane, arrivalPos="max",
                arrivalSpeed="current",
                fromTaz="", toTaz="", line="", personCapacity=0,
                personNumber=0
            )
        except TraCIException as e:
            logger.error(e, exc_info=True)
        if max_speed is not None:
            self.vehicledomain.setMaxSpeed(sumo_id, max_speed)
        self.vehicledomain.setVehicleClass(sumo_id, vehicle_class)
        self.vehicledomain.setParameter(sumo_id, 'impatience', 0.0)
        self.vehicledomain.setParameter(sumo_id, 'jmTimegapMinor', 0.5)
        self.vehicledomain.setParameter(sumo_id, 'jmIgnoreFoeSpeed', 1.0)
        self.vehicledomain.setParameter(sumo_id, 'jmIgnoreFoeProb', 1.0)

        
        return sumo_id

    def _add_ego_vehicle(self, ego_vehicle: EgoVehicle):
        """
        Adds a ego vehicle to the current ego vehicle set.

        :param ego_vehicle: the ego vehicle to be added.
        """
        if ego_vehicle.id in self._ego_vehicles:
            logger.error(
                f'Ego vehicle with id {ego_vehicle.id} already exists!',
                exc_info=True)

        self._ego_vehicles[ego_vehicle.id] = ego_vehicle

    @property
    def traci_label(self):
        """Unique label to identify simulation"""
        if self._traci_label is None:
            self._traci_label = f"{self.conf.scenario_name}-{str(time.perf_counter_ns())}-{str(uuid.uuid4())}"
        return self._traci_label

    @property
    def connected(self) -> bool:
        try:
            self._traci.getConnection(self.traci_label)
            return True
        except:
            return False

    @property
    def has_ego_vehicle(self) -> bool:
        return len(self._ego_vehicles) > 0

    @property
    def ego_vehicles(self) -> Dict[int, EgoVehicle]:
        """
        Returns the ego vehicles of the current simulation.

        """
        return self._ego_vehicles

    @ego_vehicles.setter
    def ego_vehicles(self, ego_vehicles: Dict[int, EgoVehicle]):
        """
        Sets the ego vehicles of the current simulation.

        :param ego_vehicles: ego vehicles used to set up the current simulation.
        """
        self._ego_vehicles = ego_vehicles

    @property
    def has_ego_vehicle(self) -> bool:
        return len(self._ego_vehicles) > 0

    @property
    def current_time_step(self) -> int:
        """
        :return: current time step of interface
        """
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, current_time_step: int) -> None:
        self._current_time_step = current_time_step

    def commonroad_scenario_at_time_step(self, time_step: int, add_ego=False, start_0=True) -> Scenario:
        """
        Creates and returns a commonroad scenario at the given time_step. Initial time_step=0 for all obstacles.

        :param time_step: the scenario will be created according this time step.
        :param add_ego: whether to add ego vehicles to the scenario.
        :param start_0: if set to true, initial time step of vehicles is 0, otherwise, the current time step


        """

        self.cr_scenario = Scenario(self.dt, ScenarioID.from_benchmark_id(self.conf.scenario_name, '2020a'))
        self.cr_scenario.lanelet_network = self.scenarios.lanelet_network

        # remove old obstacles from lanes
        # this is only necessary if obstacles are added to lanelets
        for lanelet in self.cr_scenario.lanelet_network.lanelets:
            lanelet.dynamic_obstacles_on_lanelet = {}

        self.cr_scenario.add_objects(self._get_cr_obstacles_at_time(time_step, add_ego=add_ego, start_0=start_0))
        return self.cr_scenario

    def commonroad_scenarios_all_time_steps(self) -> Scenario:
        """
        Creates and returns a commonroad scenario with all the dynamic obstacles.
        :param lanelet_network:
        :return: list of cr scenarios, list of cr scenarios with ego, list of planning problem sets)
        """
        self.cr_scenario = Scenario(self.dt, self.conf.scenario_name)
        self.cr_scenario.lanelet_network = self.scenarios.lanelet_network

        self.cr_scenario.add_objects(self._get_cr_obstacles_all())
        return self.cr_scenario

    def simulate_step(self, silent: bool = False) -> None:
        """
        Executes next simulation step (consisting of delta_steps sub-steps with dt_sumo=dt/delta_steps) in SUMO

        """

        if not self.initialized:
            raise ValueError("_CRSumoSimulation is not initialized")

        # simulate sumo scenario for delta_steps time steps
        for i in range(self.delta_steps):
            # send ego vehicles to SUMO
            if not self.dummy_ego_simulation and len(self.ego_vehicles) > 0:
                self._send_ego_vehicles(self.ego_vehicles, i)

            # execute SUMO simulation step
            self._traci.simulationStep()
            for ego_veh in list(self.ego_vehicles.values()):
                if not silent:
                    ego_veh._current_time_step += 1

        # remove stuck vehicles
        for sumo_id in self.vehicledomain.getIDList():
            sumo_lane_id = self.vehicledomain.getLaneID(sumo_id)
            if sumo_lane_id == '':
                continue
            if sumo_lane_id.startswith(':'):
                continue
            lanelet_id = self._map_converter.lane_id2lanelet_id.get(sumo_lane_id, None)
            lane_position = self.vehicledomain.getLanePosition(sumo_id)
            if self.include_lanes is not None and lanelet_id not in self.include_lanes and lane_position > 10.0:
                self.vehicledomain.remove(sumo_id)
            # if state.velocity < 0.1:
            #     self.vehicledomain.remove(sumo_id)


        # get updated obstacles from sumo
        if not silent:
            self._current_time_step += 1
        self._fetch_sumo_vehicles(self.current_time_step)


    def presimulation_silent(self, pre_simulation_steps: int):
        """
        Simulate SUMO without synchronization of interface. Used before starting interface simulation.

        :param pre_simulation_steps: the steps of simulation which are executed before checking the existence of ego vehicles and configured simulation step.

        """
        assert self.current_time_step == 0
        assert pre_simulation_steps >= 0, f'ego_time_start={self.conf.presimulation_steps} must be >0'

        if pre_simulation_steps == 0:
            return

        self._silent = True
        for i in range(pre_simulation_steps * self.delta_steps):
            self._traci.simulationStep()

        self._fetch_sumo_vehicles(self.current_time_step)
        self._silent = False

    @property
    def current_vehicle_ids(self) -> Set[int]:
        return set(self.obstacle_states[self.current_time_step].keys())

    def _fetch_sumo_vehicles(self, time_step: int):
        """
        Gets and stores all vehicle states from SUMO. Initializes ego vehicles when they enter simulation.

        """
        vehicle_ids = self.vehicledomain.getIDList()
        if not vehicle_ids:
            return

        for veh_id in vehicle_ids:
            state = self._get_current_state_from_sumo(self.vehicledomain, str(veh_id), SUMO_VEHICLE_PREFIX)

            # initializes new vehicle
            if veh_id not in self.ids_sumo2cr[SUMO_VEHICLE_PREFIX]:
                if veh_id.startswith(EGO_ID_START) and not self._silent:
                    # new ego vehicle
                    cr_id = self._create_cr_id(EGO_ID_START, veh_id, SUMO_VEHICLE_PREFIX)
                    if self.dummy_ego_simulation:
                        state.time_step = time_step - 1
                    else:
                        state.time_step = time_step
                    self._add_ego_vehicle(EgoVehicle(cr_id, state, self.conf.delta_steps,
                                                     self.conf.ego_veh_width,
                                                     self.conf.ego_veh_length))
                elif not self._silent:
                    # new obstacle vehicle
                    cr_id = self._create_cr_id('obstacleVehicle', veh_id, SUMO_VEHICLE_PREFIX)
                    vehicle_class = VEHICLE_TYPE_SUMO2CR[self.vehicledomain.getVehicleClass(str(veh_id))]
                    shape = self._set_veh_params(veh_id, vehicle_class)

                    self.obstacle_types[cr_id] = vehicle_class
                    self.obstacle_shapes[cr_id] = shape
                    self.obstacle_states[time_step][self.ids_sumo2cr['obstacleVehicle'][veh_id]] = state
                    self.obstacle_trajectories[self.ids_sumo2cr['obstacleVehicle'][veh_id]] = [state]
            elif veh_id in self.ids_sumo2cr['obstacleVehicle']:
                # get obstacle vehicle state
                self.obstacle_states[time_step][self.ids_sumo2cr['obstacleVehicle'][veh_id]] = state
                self.obstacle_trajectories[self.ids_sumo2cr['obstacleVehicle'][veh_id]].append(state)
            elif not self._silent and veh_id not in self.ids_sumo2cr['egoVehicle']:
                raise NotImplemented()

            # read signal state
            if not self._silent:
                signal_states = get_signal_state(self.vehicledomain.getSignals(veh_id), self.current_time_step)
                key = self.ids_sumo2cr['obstacleVehicle'][veh_id] \
                    if veh_id in self.ids_sumo2cr['obstacleVehicle'] else self.ids_sumo2cr[EGO_ID_START][veh_id]
                self.signal_states[key].append(signal_states)

            """For testing with dummy_ego_simulation"""
            if not self._silent and self.dummy_ego_simulation and veh_id in self.ids_sumo2cr['egoVehicle']:
                ego_veh = self.ego_vehicles[self.ids_sumo2cr['egoVehicle'][veh_id]]
                ori = state.orientation
                state_list = []
                for t in range(0, self.conf.delta_steps):
                    state_tmp = copy.deepcopy(state)
                    state_tmp.position = state.position + (t + 1) * state.velocity \
                                         * self.dt * np.array([np.cos(ori), np.sin(ori)])
                    state_tmp.time_step = t + 1
                    state_list.append(state_tmp)

                ego_veh.set_planned_trajectory(state_list)

    def _set_veh_params(
        self, 
        veh_id: int,
        vehicle_class: ObstacleType
    ) -> Rectangle:
        """
        :param veh_id: vehicle ID
        :param vehicle_class: vehicle class
        :return:
        """

        width = sample(self.conf.veh_params["width"][vehicle_class])
        length = sample(self.conf.veh_params["length"][vehicle_class])
        min_gap = sample(self.conf.veh_params["minGap"][vehicle_class])
        accel = sample(self.conf.veh_params["accel"][vehicle_class])
        decel = sample(self.conf.veh_params["decel"][vehicle_class])
        max_speed = sample(self.conf.veh_params["maxSpeed"][vehicle_class])
        impatience = sample(self.conf.veh_params["sigma"][vehicle_class])
        speed_factor = sample(self.conf.veh_params["speedFactor"][vehicle_class])

        self.vehicledomain.setWidth(veh_id, width)
        self.vehicledomain.setLength(veh_id, length)
        self.vehicledomain.setMinGap(veh_id, min_gap)
        self.vehicledomain.setAccel(veh_id, accel)
        self.vehicledomain.setDecel(veh_id, decel)
        self.vehicledomain.setMaxSpeed(veh_id, max_speed)
        self.vehicledomain.setImperfection(veh_id, impatience)
        self.vehicledomain.setSpeedFactor(veh_id, speed_factor)

        return Rectangle(length, width)

    def _get_current_state_from_sumo(self, domain, veh_id: str,
                                     sumo_prefix: str) -> State:
        """
        Gets the current state from sumo.
        :param domain
        :type: Union[PersonDomain, VehicleDomain]
        :param veh_id: the id of the vehicle, whose state will be returned from SUMO.

        :return: the state of the given vehicle
        """
        unique_id = sumo_prefix + veh_id
        position = np.array(domain.getPosition(veh_id))
        velocity = domain.getSpeed(veh_id)
        velocity_y = domain.getLateralSpeed(veh_id)

        cr_id = sumo2cr(veh_id, self.ids_sumo2cr)
        if cr_id and cr_id in self.obstacle_shapes:
            length = self.obstacle_shapes[cr_id].length
        else:
            length = domain.getLength(veh_id)

        if self.conf.compute_orientation \
                and self.current_time_step > 1 \
                and velocity > 0.5 \
                and unique_id in self.cached_position:
            delta_pos = position - self.cached_position[unique_id]
            orientation = math.atan2(delta_pos[1], delta_pos[0])
        else:
            orientation = math.radians(-domain.getAngle(veh_id) + 90)

        self.cached_position[unique_id] = position
        position -= 0.5 * length * np.array([np.cos(orientation), np.sin(orientation)])

        try:
            acceleration = domain.getAcceleration(veh_id)
        except AttributeError:
            acceleration = 0.0

        state = State(position=position,
                     orientation=orientation,
                     velocity=velocity,
                     velocity_y=velocity_y,
                     acceleration=acceleration,
                     time_step=self.current_time_step)

        return state

    def _get_cr_obstacles_at_time(
        self,
        time_step: int,
        add_ego: bool = False,
        start_0: bool = False,
        auto_assign_lanelet: bool = False
    ) -> List[DynamicObstacle]:
        """
        Gets current state of all vehicles in commonroad format from recorded simulation.

        :param time_step: time step of scenario
        :param add_ego: if True, add ego vehicles as well
        :param start_0: if True, initial time step of vehicles is 0, otherwise, the current time step

        """

        from traci.exceptions import TraCIException

        vehicle_dict: Dict[int, State] = self.obstacle_states[time_step]
        obstacles: List[DynamicObstacle] = []

        for veh_id, state in vehicle_dict.items():
            if start_0:
                state.time_step = 0
            else:
                state.time_step = time_step

            obstacle_type = self.obstacle_types[veh_id]
            signal_states = self.signal_states[veh_id]

            if auto_assign_lanelet:
                sumo_id = self.ids_cr2sumo['obstacleVehicle'][veh_id]
                if sumo_id not in self.vehicledomain.getIDList():
                    continue
                try:
                    lane_position = self.vehicledomain.getLanePosition(sumo_id)
                except TraCIException:
                    continue
                sumo_lane_id = self.vehicledomain.getLaneID(sumo_id)
                if sumo_lane_id == '':
                    continue
                if sumo_lane_id.startswith(':'):
                    sumo_lane_id = self.obstacle_lane_ids.get(sumo_id, None)
                    if sumo_lane_id is not None:
                        lane_position += self.lanedomain.getLength(sumo_lane_id)
                else:
                    self.obstacle_lane_ids[sumo_id] = sumo_lane_id
                lanelet_id = self._map_converter.lane_id2lanelet_id.get(sumo_lane_id, None)

                prediction = TrajectoryPrediction(
                    trajectory=Trajectory(initial_time_step=0, state_list=[state]),
                    shape=self.obstacle_shapes[veh_id],
                    center_lanelet_assignment={state.time_step: lanelet_id}
                )
            else:
                prediction = TrajectoryPrediction(
                    trajectory=Trajectory(initial_time_step=0, state_list=[state]),
                    shape=self.obstacle_shapes[veh_id],
                    center_lanelet_assignment=None
                )


            dynamic_obstacle = DynamicObstacle(
                obstacle_id=veh_id,
                obstacle_type=obstacle_type,
                initial_state=state,
                obstacle_shape=self.obstacle_shapes[veh_id],
                initial_center_lanelet_ids=None,
                initial_shape_lanelet_ids=None,
                initial_signal_state=signal_states[-1] if signal_states else None,
                signal_series=signal_states if signal_states else None,
                prediction=prediction
            )
            obstacles.append(dynamic_obstacle)

        if add_ego:
            obstacles.extend(self.get_ego_obstacles(time_step))
        return obstacles

    def _get_cr_obstacles_all(self) -> List[DynamicObstacle]:
        """
        For all recorded time steps, get states of all obstacles and convert them into commonroad dynamic obstacles. :return: list of dynamic obstacles
        """
        # transform self.obstacle_states:Dict[time_step:[veh_id:State]] to veh_state:Dict[veh_id:[time_step:State]]
        veh_state = {}
        for time_step, veh_dicts in self.obstacle_states.items():
            for veh_id, state in veh_dicts.items():
                veh_state[veh_id] = {}

        for time_step, veh_dicts in self.obstacle_states.items():
            for veh_id, state in veh_dicts.items():
                state.time_step = time_step
                veh_state[veh_id][time_step] = state

        # get all vehicles' ids for id conflict check between lanelet_id and veh_id
        self.veh_ids: List = [*veh_state]

        # create cr obstacles
        obstacles = []
        for veh_id, time_dict in veh_state.items():
            state_list = list(time_dict.values())
            # coordinate transformation for all positions from sumo format to commonroad format
            obstacle_shape = self.obstacle_shapes[veh_id]
            obstacle_type = self.obstacle_types[veh_id]
            initial_state = state_list[0]

            assert self.conf.video_start != self.conf.video_end, \
                "Simulation start time and end time are the same. Please set simulation interval."

            if len(state_list) > 4:
                obstacle_trajectory = Trajectory(state_list[0].time_step, state_list[0:])
                obstacle_prediction = TrajectoryPrediction(obstacle_trajectory, obstacle_shape)
                center_lanelets = None
                shape_lanelets = None
                if self.conf.add_lanelets_to_dyn_obstacles:
                    center_lanelets = {lanelet_id for lanelet_ids in
                                       self.cr_scenario.lanelet_network.find_lanelet_by_position(
                                           [initial_state.position])
                                       for lanelet_id in lanelet_ids}
                    shape_lanelets = {lanelet_id for lanelet_ids in
                                      self.cr_scenario.lanelet_network.find_lanelet_by_position(
                                          [initial_state.position + v
                                           for v in self.obstacle_shapes[veh_id].vertices])
                                      for lanelet_id in lanelet_ids}

                signal_states = self.signal_states[veh_id]
                dynamic_obstacle = DynamicObstacle(
                    obstacle_id=veh_id,
                    obstacle_type=obstacle_type,
                    initial_state=initial_state,
                    obstacle_shape=obstacle_shape,
                    prediction=obstacle_prediction,
                    initial_center_lanelet_ids=center_lanelets if center_lanelets else None,
                    initial_shape_lanelet_ids=shape_lanelets if shape_lanelets else None,
                    initial_signal_state=signal_states[0] if signal_states else None,
                    signal_series=signal_states[1:] if signal_states else None)  # add a trajectory element
                obstacles.append(dynamic_obstacle)
            else:
                pass
                # logger.debug(f'Vehicle {veh_id} has been simulated less than 5 time steps. Not converted to cr obstacle.')
        return obstacles

    @lru_cache()
    def _get_ids_of_map(self) -> Set[int]:
        """
        Get a list of ids of all the lanelets from the cr map which is converted from a osm map.
        :return: list of lanelets' ids
        """
        ids = set()
        for lanelet in self.scenarios.lanelet_network.lanelets:
            ids.add(lanelet.lanelet_id)
        for ts in self.scenarios.lanelet_network.traffic_signs:
            ids.add(ts.traffic_sign_id)
        for ts in self.scenarios.lanelet_network.traffic_lights:
            ids.add(ts.traffic_light_id)
        for ts in self.scenarios.lanelet_network.intersections:
            ids.add(ts.intersection_id)
            for inc in ts.incomings:
                ids.add(inc.incoming_id)
        return ids

    def get_ego_obstacles(self, time_step: Union[int, None] = None) -> List[DynamicObstacle]:
        """
        Get list of ego vehicles converted to Dynamic obstacles
        :param time_step: initial time step, if None, get complete driven trajectory
        :return:
        """
        obstacles = []
        for veh_id, ego_veh in self.ego_vehicles.items():
            obs = ego_veh.get_dynamic_obstacle(time_step)
            if obs is not None:
                obstacles.append(obs)

        return obstacles

    def _send_ego_vehicles(self, ego_vehicles: Dict[int, EgoVehicle], delta_step: int = 0) -> None:
        """
        Sends the information of ego vehicles to SUMO.

        :param ego_vehicles: list of dictionaries.
            For each ego_vehicle, write tuple (cr_ego_id, cr_position, cr_lanelet_id, cr_orientation, cr_lanelet_id)
            cr_lanelet_id can be omitted but this is not recommended, if the lanelet is known for sure.
        :param delta_step: which time step of the planned trajectory should be sent

        """
        for id_cr, ego_vehicle in ego_vehicles.items():
            assert ego_vehicle.current_time_step == self.current_time_step, \
                f'Trajectory of ego vehicle has not been updated. Still at time_step {ego_vehicle.current_time_step}, ' \
                f'while simulation step {self.current_time_step + 1} should be simulated.'

            planned_trajectory = ego_vehicle.get_planned_trajectory

            if planned_trajectory:
                # Check if there is a lanelet change in the configured time window
                lc_future_status, lc_duration = self.check_lanelets_future_change(
                    ego_vehicle.current_state, planned_trajectory)
                # If there is a lanelet change, check whether the change is just started
                lc_status = self._check_lc_start(id_cr, lc_future_status)
                # Calculate the sync mechanism based on the future information
                sync_mechanism = self._check_sync_mechanism(
                    lc_status, id_cr, ego_vehicle.current_state)
                # Execute MoveXY or SUMO lane change according to the sync mechanism
                planned_state = ego_vehicle.get_planned_state(delta_step)
                self.forward_info2sumo(planned_state, sync_mechanism, lc_duration,
                                        id_cr)

    def _get_ego_ids(self) -> Dict[int, str]:
        """
        Returns a dictionary with all current ego vehicle ids and corresponding sumo ids
        """
        return self.ids_cr2sumo[EGO_ID_START]

    def _create_sumo_id(self, list_ids_ego_used: List[int]) -> int:
        """
        Generates a new unused id for SUMO
        :return:
        """
        id_list = self.vehicledomain.getIDList()
        new_id = int(len(id_list))
        i = 0
        while i < 1000:
            if str(new_id) not in id_list and new_id not in list_ids_ego_used:
                return new_id
            else:
                new_id += 1
                i += 1

    def _create_cr_id(self, type: str, sumo_id: str, sumo_prefix: str, cr_id: int = None) -> int:
        """
        Generates a new cr ID and adds it to ID dictionaries

        :param type: one of the keys in params.id_convention; the type defines the first digit of the cr_id
        :param sumo_id: id in sumo simulation
        :param sumo_prefix: str giving what set of sumo ids to use

        :return: cr_id as int
        """
        cr_id = generate_cr_id(type, sumo_id, sumo_prefix, self.ids_sumo2cr, self._max_cr_id)

        self.ids_sumo2cr[type][sumo_id] = cr_id
        self.ids_sumo2cr[sumo_prefix][sumo_id] = cr_id
        self.ids_cr2sumo[type][cr_id] = sumo_id
        self.ids_cr2sumo[sumo_prefix][cr_id] = sumo_id

        self._max_cr_id = max(self._max_cr_id, cr_id)
        return cr_id

    @property
    def _silent(self) -> bool:
        """Ego vehicle is not synced in this mode."""
        return self.__silent

    @_silent.setter
    def _silent(self, silent) -> None:
        assert self.current_time_step == 0
        self.__silent = silent

    def stop(self) -> None:
        if not self._running:
            return
        """ Exits SUMO Simulation"""
        try:
            self._traci.close()
        except:
            pass
        self.reset_variables()
        self._running = False
        #sys.stdout.flush()

    def __del__(self) -> None:
        self.stop()

    # Ego sync functions
    def check_lanelets_future_change(
            self, current_state: State,
            planned_traj: List[State]) -> Tuple[str, int]:
        """
        Checks the lanelet changes of the ego vehicle in the future time_window.

        :param lanelet_network: object of the lanelet network
        :param time_window: the time of the window to check the lanelet change
        :param traj_index: index of the planner output corresponding to the current time step

        :return: lc_status, lc_duration: lc_status is the status of the lanelet change in the next time_window; lc_duration is the unit of time steps (using sumo dt)

        """
        lc_duration_max = min(self.conf.lanelet_check_time_window,
                              len(planned_traj))
        lanelet_network = self.scenarios.lanelet_network
        lc_status = 'NO_LC'
        lc_duration = 0

        # find current lanelets
        current_position = current_state.position
        current_lanelets_ids = lanelet_network.find_lanelet_by_position([current_position])[0]
        current_lanelets = [
            find_lanelet_by_id(lanelet_network, id)
            for id in current_lanelets_ids
        ]

        # check for lane change
        for current_lanelet in current_lanelets:
            for t in range(lc_duration_max):
                future_lanelet_ids = lanelet_network.find_lanelet_by_position(
                    [planned_traj[t].position])[0]
                if current_lanelet.adj_right in future_lanelet_ids:
                    lc_status = 'RIGHT_LC'
                    lc_duration = 2 * t * self.conf.delta_steps
                    break
                elif current_lanelet.adj_left in future_lanelet_ids:
                    lc_status = 'LEFT_LC'
                    lc_duration = 2 * t * self.conf.delta_steps
                    break
                else:
                    pass

        # logger.debug('current lanelets: ' + str(current_lanelets))
        # logger.debug('lc_status=' + lc_status)
        # logger.debug('lc_duration=' + str(lc_duration))
        return lc_status, lc_duration

    def _check_lc_start(self, ego_id: str, lc_future_status: str) -> str:
        """
        This function checks if a lane change is started according to the change in the lateral position and the lanelet
        change prediction. Note that checking the change of lateral position only is sensitive to the tiny changes, also
        at the boundaries of the lanes the lateral position sign is changed because it will be calculated relative to
        the new lane. So we check the future lanelet change to avoid these issues.

        :param ego_id: id of the ego vehicle
        :param lc_future_status: status of the future lanelet changes

        :return: lc_status: the status whether the ego vehicle starts a lane change or no
        """
        lateral_position = self.vehicledomain.getLateralLanePosition(
            cr2sumo(ego_id, self.ids_cr2sumo))

        if lc_future_status == 'NO_LC' or not id in self.lateral_position_buffer:
            lc_status = 'NO_LC'
        elif lc_future_status == 'RIGHT_LC' \
            and self.lateral_position_buffer[id] > self.conf.lane_change_tol + lateral_position:
            lc_status = 'RIGHT_LC_STARTED'
        elif lc_future_status == 'LEFT_LC' \
            and self.lateral_position_buffer[id] < -self.conf.lane_change_tol + lateral_position:
            lc_status = 'LEFT_LC_STARTED'
        else:
            lc_status = 'NO_LC'

        # logger.debug('LC current status: ' + lc_status)

        self.lateral_position_buffer[id] = lateral_position

        return lc_status

    def _consistency_protection(self, ego_id: str,
                                current_state: State) -> str:
        """
        Checks the L2 distance between SUMO position and the planner position and returns CONSISTENCY_ERROR if it is
        above the configured margin.

        :param ego_id: id of the ego vehicle (string)
        :param current_state: the current state read from the commonroad motion planner

        :return: retval: the status whether there is a consistency error between sumo and planner positions or not
        """
        cons_error = 'CONSISTENCY_NO_ERROR'

        pos_sumo = self.vehicledomain.getPosition(cr2sumo(ego_id, self.ids_cr2sumo))
        pos_cr = current_state.position
        dist_error = np.linalg.norm(pos_cr - pos_sumo)
        if dist_error > self.conf.protection_margin:
            cons_error = 'CONSISTENCY_ERROR'

        # logger.debug('SUMO X: ' + str(pos_sumo[0]) + ' **** SUMO Y: ' + str(pos_sumo[1]))
        # logger.debug('TRAJ X: ' + str(pos_cr[0]) + ' **** TRAJ Y: ' + str(pos_cr[1]))
        # logger.debug('Error Value: ' + str(dist_error))
        # logger.debug('Error Status: ' + cons_error)

        if self._lc_inaction == 0:
            cons_error = 'CONSISTENCY_NO_ERROR'
        return cons_error

    def _check_sync_mechanism(self, lc_status: str, ego_id: int,
                              current_state: State) -> str:
        """
        Defines the sync mechanism type that should be executed according to the ego vehicle motion.

        :param lc_status: status of the lanelet change in the next time_window
        :param ego_id: id of the ego vehicle (string)
        :param current_state: the current state read from the commonroad motion planner

        :return: retval: the sync mechanism that should be followed while communicating from the interface to sumo
        """
        if self.conf.lane_change_sync == True:
            # Check the error between SUMO and CR positions
            cons_error = self._consistency_protection(ego_id, current_state)
            if cons_error == 'CONSISTENCY_NO_ERROR':  # CONSISTENCY_NO_ERROR means error below the configured margin
                if self._lc_inaction == 0:
                    if lc_status == 'RIGHT_LC_STARTED':
                        self._lc_inaction = 1
                        retval = 'SYNC_SUMO_R_LC'
                    elif lc_status == 'LEFT_LC_STARTED':
                        self._lc_inaction = 1
                        retval = 'SYNC_SUMO_L_LC'
                    else:
                        retval = 'SYNC_MOVE_XY'
                else:  # There is a lane change currently in action so do nothing and just increament the counter
                    self._lc_counter += 1
                    if self._lc_counter >= self._lc_duration_max:
                        self._lc_counter = 0
                        self._lc_inaction = 0
                    retval = 'SYNC_DO_NOTHING'
            else:  # There is a consistency error so force the sync mechanism to moveToXY to return back to zero error
                retval = 'SYNC_MOVE_XY'
                self._lc_counter = 0
                self._lc_inaction = 0
        else:
            retval = 'SYNC_MOVE_XY'

        # logger.debug('Sync Mechanism is: ' + retval)
        # logger.debug('Lane change performed since ' + str(self._lc_counter))
        return retval

    def forward_info2sumo(self, planned_state: State, sync_mechanism: str, lc_duration: int, ego_id: int = None):
        """
        Forwards the information to sumo (either initiate moveToXY or changeLane) according to the sync mechanism.

        :param planned_state: the planned state from commonroad motion planner
        :param sync_mechanism: the sync mechanism that should be followed while communicating from the interface to sumo
        :param lc_duration: lane change duration, expressed in number of time steps
        :param ego_id: id of the ego vehicle, will assume first ego vehicle if None is provided
        """

        ego_id = ego_id or self.ego_vehicle_first.id

        id_sumo = cr2sumo(ego_id, self.ids_cr2sumo)

        if sync_mechanism == 'SYNC_MOVE_XY':
            len_half = 0.5 * self.ego_vehicles[ego_id].length
            position_sumo = [0, 0]
            position_sumo[0] = planned_state.position[0] + len_half * math.cos(planned_state.orientation)
            position_sumo[1] = planned_state.position[1] + len_half * math.sin(planned_state.orientation)
            sumo_angle = 90 - math.degrees(planned_state.orientation)

            if not id_sumo in self.vehicledomain.getIDList():
                from traci.exceptions import TraCIException
                try:
                    self._add_vehicle_to_sim(id_sumo)
                except TraCIException:
                    pass
            self.vehicledomain.moveToXY(vehID=id_sumo,
                                        edgeID='dummy',
                                        lane=-1,
                                        x=position_sumo[0],
                                        y=position_sumo[1],
                                        angle=sumo_angle,
                                        keepRoute=2)

            self.vehicledomain.setSpeedMode(id_sumo, 0)
            self.vehicledomain.setSpeed(id_sumo, planned_state.velocity)

        elif sync_mechanism == 'SYNC_SUMO_R_LC':
            # A lane change (right lane change) is just started, so we will initiate lane change request by traci
            # self.vehicledomain.setLaneChangeDuration(cr2sumo(default_ego_id, self.ids_cr2sumo), lc_duration)
            self.vehicledomain.setLaneChangeMode(id_sumo, 512)
            targetlane = self.vehicledomain.getLaneIndex(id_sumo) - 1
            self.vehicledomain.changeLane(id_sumo, targetlane, lc_duration)
        elif sync_mechanism == 'SYNC_SUMO_L_LC':
            # A lane change (left lane change) is just started, so we will initiate lane change request by traci
            # self.vehicledomain.setLaneChangeDuration(cr2sumo(default_ego_id, self.ids_cr2sumo), lc_duration)
            self.vehicledomain.setLaneChangeMode(id_sumo, 512)
            targetlane = self.vehicledomain.getLaneIndex(id_sumo) + 1
            self.vehicledomain.changeLane(id_sumo, targetlane, lc_duration)
        elif sync_mechanism == 'SYNC_DO_NOTHING':
            pass
        else:
            pass
