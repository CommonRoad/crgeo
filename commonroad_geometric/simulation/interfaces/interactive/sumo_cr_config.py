import os
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from commonroad.common.util import Interval
from commonroad.scenario.obstacle import ObstacleType

from commonroad_geometric.simulation.interfaces.interactive.sumo_commandline_config import SumoCommandLineConfig

# TODO Unused
@dataclass
class SumoInitializationConfig():
    ego_id_start:           str = 'egoVehicle'
    sumo_vehicle_prefix:    str = 'vehicle'
    sumo_pedestrian_prefix: str = 'pedestrian'

    cl_conf: SumoCommandLineConfig = SumoCommandLineConfig()

    id_dict:                Dict[str, int] = field(default_factory=lambda: {'obstacleVehicle': 3, 'egoVehicle': 4, 'obstaclePedestrian': 5})

    # logging level for logging module
    logging_level:          str = 'ERROR'  # select DEBUG, INFO, WARNING, ERROR, CRITICAL               # _cr_sumo_simulation._init_logging()

    ##
    ## simulation
    ##

    # number of sub-steps simulated in SUMO during every dt
    delta_steps:                    int = 1

    ## _cr_sumo_simulation.dt_sumo == dt / delta_steps == sumo_commandline_config.step_length
    # length of simulation step of the interface
    dt:                             float = delta_steps * (cl_conf.step_length if cl_conf.step_length is not None else 1)

    # number of time steps before simulation with ego vehicle starts
    presimulation_steps:            int = 30                            # _cr_sumo_simulation.presimulation_silent()

    # number of simulated (and synchronized) time steps
    simulation_steps                 = 1e8                              # TODO: unused apart of self.video_end
    with_sumo_gui:                  bool = False                        # _cr_sumo_simulation.initialize(), TODO: for what?
    # re-compute orientation when fetching vehicles from SUMO.
    # Avoids lateral "sliding" at lane changes at computational costs
    compute_orientation:            bool = True                         # _cr_sumo_simulation._get_current_state_from_sumo()
    add_lanelets_to_dyn_obstacles:  bool = False                        # _cr_sumo_simulation._get_cr_obstacles_all()
    random_seed_trip_generation:    int = 0                             #TODO: unused
    highway_mode:                   bool = False                        #TODO: unused

    # ego vehicle
    ego_veh_width:                  float = 1.6                         # _cr_sumo_simulation.init_ego_vehicle(), _cr_sumo_simulation._fetch_sumo_vehicles()
    ego_veh_length:                 float = 4.3                         # _cr_sumo_simulation.init_ego_vehicle(), _cr_sumo_simulation._fetch_sumo_vehicles()

    ##
    ## Plotting
    ##
    video_start:                    int = 1                             # _cr_sumo_simulation._get_cr_obstacles_all()
    video_end:                      float = simulation_steps            # _cr_sumo_simulation._get_cr_obstacles_all()

    ##
    ## ego vehicle sync parameters
    ##
    # Time window to detect the lanelet change in seconds
    lanelet_check_time_window:  int = int(2 / dt)                                                       # _cr_sumo_simulation.check_lanelets_future_change()
    # The absolute margin allowed between the planner position and ego position in SUMO
    protection_margin:          float = 2.0                                                             # _cr_sumo_simulation._consistency_protection()
    # Variable can be used  to force the consistency to certain number of steps
    consistency_window:         int = 4                                                                 # TODO: no references
    # Used to limit the sync mechanism only to move xy
    lane_change_sync:           bool = False                                                            # _cr_sumo_simulation._check_sync_mechanism()
    # tolerance for detecting start of lane change
    lane_change_tol:            float = 0.00                                                            # _cr_sumo_simulation._check_lc_start()

    ##
    ## TRAFFIC GENERATION
    ##

    ## random_seed == sumo_comandline_config.seed
    # random seed for deterministic sumo traffic generation (applies if not set to None)
    random_seed:                Optional[int] = cl_conf.seed                                            # TODO: replace with sumo_comandline_config.seed

    # other vehicles size bound (values are sampled from normal distribution within bounds)
    vehicle_length_interval:    float = 0.4                                                             # TODO: no references
    vehicle_width_interval:     float = 0.2                                                             # TODO: no references

    # probability distribution of different vehicle classes. Do not need to sum up to 1.
    veh_distribution: Dict[ObstacleType, float] = field(default_factory=lambda: {                                                     # _cr_sumo_simulation.initialize()
        ObstacleType.CAR: 12,
        ObstacleType.TRUCK: 0.8,
        ObstacleType.BUS: 0.3,
        ObstacleType.BICYCLE: 0.0,
        ObstacleType.PEDESTRIAN: 0
    })

    # default vehicle attributes to determine edge restrictions

    # vehicle attributes
    veh_params: Dict[str, Dict[ObstacleType, Union[Interval, float]]] = field(default_factory=lambda: {               # _cr_sumo_simulation._set_veh_params()
        # maximum length
        'length': {
            ObstacleType.CAR: Interval(4.0, 5.0),
            ObstacleType.TRUCK: Interval(8.5, 12.5),
            ObstacleType.BUS: Interval(8.8, 14.4),
            ObstacleType.BICYCLE: 2.,
            ObstacleType.PEDESTRIAN: 0.415
        },
        # maximum width
        'width': {
            ObstacleType.CAR: Interval(1.8, 1.9),
            ObstacleType.TRUCK: Interval(2.5, 2.6),
            ObstacleType.BUS: Interval(2.6, 2.7),
            ObstacleType.BICYCLE: 0.68,
            ObstacleType.PEDESTRIAN: 0.678
        },
        'minGap': {
            ObstacleType.CAR: 2.5,
            ObstacleType.TRUCK: 2.5,
            ObstacleType.BUS: 2.5,
            ObstacleType.BICYCLE: 1.,
            ObstacleType.PEDESTRIAN: 0.25
        },
        # the following values cannot be set of pedestrians
        'accel': {
            # default 2.9 m/s²
            ObstacleType.CAR: Interval(2, 2.9),
            # default 1.3
            ObstacleType.TRUCK: Interval(1, 1.5),
            # default 1.2
            ObstacleType.BUS: Interval(1, 1.4),
            # default 1.2
            ObstacleType.BICYCLE: Interval(1, 1.4),
        },
        'decel': {
            # default 7.5 m/s²
            ObstacleType.CAR: Interval(4, 6.5),
            # default 4
            ObstacleType.TRUCK: Interval(3, 4.5),
            # default 4
            ObstacleType.BUS: Interval(3, 4.5),
            # default 3
            ObstacleType.BICYCLE: Interval(2.5, 3.5),
        },
        'maxSpeed': {
            # default 180/3.6 m/s
            ObstacleType.CAR: 180 / 3.6,
            # default 130/3.6
            ObstacleType.TRUCK: 130 / 3.6,
            # default 85/3.6
            ObstacleType.BUS: 85 / 3.6,
            # default 85/3.6
            ObstacleType.BICYCLE: 25 / 3.6,
        },
        'sigma': {
            ObstacleType.CAR: Interval(0.7, 0.9),
            ObstacleType.TRUCK: Interval(0.7, 0.9),
            ObstacleType.BUS: Interval(0.7, 0.9),
            ObstacleType.BICYCLE: Interval(0.7, 0.9),
        },
        'speedFactor': {
            ObstacleType.CAR: Interval(0.9, 1.1),
            ObstacleType.TRUCK: Interval(0.9, 1.1),
            ObstacleType.BUS: Interval(0.9, 1.1),
            ObstacleType.BICYCLE:Interval(0.9, 1.1),
        },
    })

