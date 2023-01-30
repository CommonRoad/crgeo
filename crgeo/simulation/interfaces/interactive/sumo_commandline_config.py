import inspect
import os
import subprocess
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from crgeo.external.sumocr.sumo_config.pathConfig import SUMO_BINARY


@dataclass
class SumoCommandLineConfig:
    """
    This file was partially generated with the script in scripts/gen_flags_for_sumo_commandline_config.py

    representation of flags and their values for sumo call
    field names correspond to flag names
    underscores inside field names '_' are replaced with dashes '-'
    two underscores '__' are replaced with a dot '.'
    fields with underscore at the start are ignored
    when building the command

    For more information: https://sumo.dlr.de/docs/sumo.html
    """
    _sumo_binary: Optional[str] = SUMO_BINARY

    # Configurations
    configuration_file:											    Optional[str] = None
    save_configuration:											    Optional[str] = None
    save_configuration__relative:									Optional[bool] = None
    save_template:													Optional[str] = None
    save_schema:													Optional[str] = None
    save_commented:												    Optional[bool] = None

    # Input
    net_file:														Optional[str] = None
    route_files:													Optional[str] = None
    additional_files:												Optional[str] = None
    weight_files:													Optional[str] = None
    weight_attribute:												Optional[str] = None
    load_state:													    Optional[str] = None
    load_state__offset:												Optional[float] = None
    load_state__remove_vehicles:									Optional[str] = None
    junction_taz:													Optional[bool] = None

    # Output
    write_license:													Optional[bool] = None
    output_prefix:													Optional[str] = None
    precision:														Optional[int] = None
    precision__geo:													Optional[int] = None
    human_readable_time:											Optional[bool] = None
    netstate_dump:													Optional[str] = None
    netstate_dump__empty_edges:										Optional[bool] = None
    netstate_dump__precision:										Optional[int] = None
    emission_output:												Optional[str] = None
    emission_output__precision:										Optional[int] = None
    emission_output__geo:											Optional[bool] = None
    emission_output__step_scaled:									Optional[bool] = None
    battery_output:													Optional[str] = None
    battery_output__precision:										Optional[int] = None
    elechybrid_output:												Optional[str] = None
    elechybrid_output__precision:									Optional[int] = None
    elechybrid_output__aggregated:									Optional[bool] = None
    chargingstations_output:										Optional[str] = None
    overheadwiresegments_output:									Optional[str] = None
    substations_output:												Optional[str] = None
    substations_output__precision:									Optional[int] = None
    fcd_output:														Optional[str] = None
    fcd_output__geo:												Optional[bool] = None
    fcd_output__signals:											Optional[bool] = None
    fcd_output__distance:											Optional[bool] = None
    fcd_output__acceleration:										Optional[bool] = None
    fcd_output__max_leader_distance:								Optional[float] = None
    fcd_output__params:												Optional[str] = None        #Add generic parameter values to the FCD output
    fcd_output__filter_edges__input_file:							Optional[str] = None
    fcd_output__attributes:											Optional[List[str]] = None  #List attributes that should be included in the FCD output
    fcd_output__filter_shapes:										Optional[List[str]] = None  #List shape names that should be used to filter the FCD output
    device__ssm__filter_edges__input_file:							Optional[str] = None
    full_output:													Optional[str] = None
    queue_output:													Optional[str] = None
    queue_output__period:											Optional[float] = None
    vtk_output:														Optional[str] = None
    amitran_output:													Optional[str] = None
    summary_output:													Optional[str] = None
    summary_output__period:											Optional[float] = None
    person_summary_output:											Optional[str] = None
    tripinfo_output:												Optional[str] = None
    tripinfo_output__write_unfinished:								Optional[bool] = None
    tripinfo_output__write_undeparted:								Optional[bool] = None
    vehroute_output:												Optional[str] = None
    vehroute_output__exit_times:									Optional[bool] = None
    vehroute_output__last_route:									Optional[bool] = None
    vehroute_output__sorted:										Optional[bool] = None
    vehroute_output__dua:											Optional[bool] = None
    vehroute_output__cost:											Optional[bool] = None
    vehroute_output__intended_depart:								Optional[bool] = None
    vehroute_output__route_length:									Optional[bool] = None
    vehroute_output__write_unfinished:								Optional[bool] = None
    vehroute_output__skip_ptlines:									Optional[bool] = None
    vehroute_output__incomplete:									Optional[bool] = None
    vehroute_output__stop_edges:									Optional[bool] = None
    vehroute_output__speedfactor:									Optional[bool] = None
    vehroute_output__internal:										Optional[bool] = None
    personroute_output:												Optional[str] = None
    link_output:													Optional[str] = None
    railsignal_block_output:										Optional[str] = None
    bt_output:														Optional[str] = None
    lanechange_output:												Optional[str] = None
    lanechange_output__started:										Optional[bool] = None
    lanechange_output__ended:										Optional[bool] = None
    lanechange_output__xy:											Optional[bool] = None
    stop_output:													Optional[str] = None
    stop_output__write_unfinished:									Optional[bool] = None
    collision_output:												Optional[str] = None
    edgedata_output:												Optional[str] = None
    lanedata_output:												Optional[str] = None
    statistic_output:												Optional[str] = None
    save_state__times:												Optional[List[float]] = None   #Use TIME[] as times at which a network state written
    save_state__period:												Optional[float] = None
    save_state__period__keep:										Optional[int] = None
    save_state__prefix:												Optional[str] = None
    save_state__suffix:												Optional[str] = None
    save_state__files:												Optional[str] = None
    save_state__rng:												Optional[bool] = None
    save_state__transportables:										Optional[bool] = None
    save_state__constraints:										Optional[bool] = None
    save_state__precision:											Optional[int] = None

    # Time
    begin:															Optional[float] = None
    end:															Optional[float] = None
    step_length:													Optional[float] = None

    # Processing
    step_method__ballistic:											Optional[bool] = None
    extrapolate_departpos:											Optional[bool] = None
    threads:														Optional[int] = None
    lateral_resolution:												Optional[float] = 0.8
    route_steps:													Optional[float] = None
    no_internal_links:												Optional[bool] = None
    ignore_junction_blocker:										Optional[float] = 1.0
    ignore_route_errors:											Optional[bool] = None
    ignore_accidents:												Optional[bool] = None
    collision__action:												Optional[str] = None
    collision__stoptime:											Optional[float] = None
    collision__check_junctions:										Optional[bool] = None
    collision__check_junctions__mingap:								Optional[float] = None
    collision__mingap_factor:										Optional[float] = None
    max_num_vehicles:												Optional[int] = None
    max_num_teleports:												Optional[int] = None
    scale:															Optional[float] = None
    scale_suffix:													Optional[str] = None
    time_to_teleport:												Optional[float] = None
    time_to_teleport__highways:										Optional[float] = None
    time_to_teleport__highways__min_speed:							Optional[float] = None
    time_to_teleport__disconnected:									Optional[float] = None
    time_to_teleport__remove:										Optional[bool] = None
    time_to_teleport__ride:											Optional[float] = None
    time_to_teleport__bidi:											Optional[float] = None
    waiting_time_memory:											Optional[float] = 1.0
    startup_wait_threshold:											Optional[float] = None
    max_depart_delay:												Optional[float] = None
    sloppy_insert:													Optional[bool] = None
    eager_insert:													Optional[bool] = None
    emergency_insert:												Optional[bool] = None
    random_depart_offset:											Optional[float] = None
    lanechange__duration:											Optional[float] = 0.0
    lanechange__overtake_right:										Optional[bool] = None
    tls__all_off:													Optional[bool] = True
    tls__actuated__show_detectors:									Optional[bool] = None
    tls__actuated__jam_threshold:									Optional[float] = None
    tls__actuated__detector_length:									Optional[float] = None
    tls__delay_based__detector_range:								Optional[float] = None
    tls__yellow__min_decel:											Optional[float] = None
    railsignal_moving_block:										Optional[bool] = None
    time_to_impatience:												Optional[float] = 1.0
    default__action_step_length:									Optional[float] = None
    default__carfollowmodel:										Optional[str] = None
    default__speeddev:												Optional[float] = None
    default__emergencydecel:										Optional[str] = None
    overhead_wire__solver:											Optional[bool] = None
    overhead_wire__recuperation:									Optional[bool] = None
    overhead_wire__substation_current_limits:						Optional[bool] = None
    emergencydecel__warning_threshold:								Optional[float] = None
    parking__maneuver:												Optional[bool] = None
    use_stop_ended:													Optional[bool] = None
    pedestrian__model:												Optional[str] = None
    pedestrian__striping__stripe_width:								Optional[float] = None
    pedestrian__striping__dawdling:									Optional[float] = None
    pedestrian__striping__mingap_to_vehicle:						Optional[float] = None
    pedestrian__striping__jamtime:									Optional[float] = None
    pedestrian__striping__jamtime__crossing:						Optional[float] = None
    pedestrian__striping__jamtime__narrow:							Optional[float] = None
    pedestrian__striping__reserve_oncoming:							Optional[float] = None
    pedestrian__striping__reserve_oncoming__junctions:				Optional[float] = None
    pedestrian__striping__legacy_departposlat:						Optional[bool] = None
    pedestrian__striping__walkingarea_detail:						Optional[int] = None
    pedestrian__remote__address:									Optional[str] = None
    ride__stop_tolerance:											Optional[float] = None
    persontrip__walk_opposite_factor:								Optional[float] = None
    routing_algorithm:												Optional[str] = None
    weights__random_factor:											Optional[float] = None
    weights__minor_penalty:											Optional[float] = None
    weights__tls_penalty:											Optional[float] = None
    weights__priority_factor:										Optional[float] = None
    weights__separate_turns:										Optional[float] = None
    astar__all_distances:											Optional[str] = None
    astar__landmark_distances:										Optional[str] = None
    persontrip__walkfactor:											Optional[float] = None
    persontrip__transfer__car_walk:									Optional[str] = None
    persontrip__transfer__taxi_walk:								Optional[str] = None
    persontrip__transfer__walk_taxi:								Optional[str] = None
    persontrip__default__group:										Optional[str] = None
    persontrip__taxi__waiting_time:									Optional[float] = None
    railway__max_train_length:										Optional[float] = None
    replay_rerouting:												Optional[bool] = None
    device__rerouting__probability:									Optional[float] = None
    device__rerouting__explicit:									Optional[str] = None
    device__rerouting__deterministic:								Optional[bool] = None
    device__rerouting__period:										Optional[float] = None
    device__rerouting__pre_period:									Optional[float] = None
    device__rerouting__adaptation_weight:							Optional[float] = None
    device__rerouting__adaptation_steps:							Optional[int] = None
    device__rerouting__adaptation_interval:							Optional[float] = None
    device__rerouting__with_taz:									Optional[bool] = None
    device__rerouting__init_with_loaded_weights:					Optional[bool] = None
    device__rerouting__threads:										Optional[int] = None
    device__rerouting__synchronize:									Optional[bool] = None
    device__rerouting__railsignal:									Optional[bool] = None
    device__rerouting__bike_speeds:									Optional[bool] = None
    device__rerouting__output:										Optional[str] = None
    person_device__rerouting__probability:							Optional[float] = None
    person_device__rerouting__explicit:								Optional[str] = None
    person_device__rerouting__deterministic:						Optional[bool] = None
    person_device__rerouting__period:								Optional[float] = None

    # Report
    verbose:														Optional[bool] = False
    print_options:													Optional[bool] = None
    help:															Optional[bool] = None
    version:														Optional[bool] = None
    xml_validation:													Optional[str] = None
    xml_validation__net:											Optional[str] = None
    xml_validation__routes:											Optional[str] = None
    no_warnings:													Optional[bool] = True
    aggregate_warnings:												Optional[int] = None
    log:															Optional[str] = None
    message_log:													Optional[str] = None
    error_log:														Optional[str] = None
    duration_log__disable:											Optional[bool] = None
    duration_log__statistics:										Optional[bool] = None
    no_step_log:													Optional[bool] = None
    step_log__period:												Optional[int] = None

    #Emission
    emissions__volumetric_fuel:										Optional[bool] = None
    phemlight_path:													Optional[str] = None
    phemlight_year:													Optional[int] = None
    phemlight_temperature:											Optional[float] = None
    device__emissions__probability:									Optional[float] = None
    device__emissions__explicit:									Optional[str] = None
    device__emissions__deterministic:								Optional[bool] = None
    device__emissions__begin:										Optional[str] = None
    device__emissions__period:										Optional[str] = None

    # Communication
    device__btreceiver__probability:								Optional[float] = None
    device__btreceiver__explicit:									Optional[str] = None
    device__btreceiver__deterministic:								Optional[bool] = None
    device__btreceiver__range:										Optional[float] = None
    device__btreceiver__all_recognitions:							Optional[bool] = None
    device__btreceiver__offtime:									Optional[float] = None
    device__btsender__probability:									Optional[float] = None
    device__btsender__explicit:										Optional[str] = None
    device__btsender__deterministic:								Optional[bool] = None
    person_device__btsender__probability:							Optional[float] = None
    person_device__btsender__explicit:								Optional[str] = None
    person_device__btsender__deterministic:							Optional[bool] = None
    person_device__btreceiver__probability:							Optional[float] = None
    person_device__btreceiver__explicit:							Optional[str] = None
    person_device__btreceiver__deterministic:						Optional[bool] = None

    # Battery
    device__battery__probability:									Optional[float] = None
    device__battery__explicit:										Optional[str] = None
    device__battery__deterministic:									Optional[bool] = None
    device__battery__track_fuel:									Optional[bool] = None

    # Example Device
    device__example__probability:									Optional[float] = None
    device__example__explicit:										Optional[str] = None
    device__example__deterministic:									Optional[bool] = None
    device__example__parameter:										Optional[float] = None

    # Ssm Device
    device__ssm__probability:										Optional[float] = None
    device__ssm__explicit:											Optional[str] = None
    device__ssm__deterministic:										Optional[bool] = None
    device__ssm__measures:											Optional[str] = None
    device__ssm__thresholds:										Optional[str] = None
    device__ssm__trajectories:										Optional[bool] = None
    device__ssm__range:												Optional[float] = None
    device__ssm__extratime:											Optional[float] = None
    device__ssm__file:												Optional[str] = None
    device__ssm__geo:												Optional[bool] = None
    device__ssm__write_positions:									Optional[bool] = None
    device__ssm__write_lane_positions:								Optional[bool] = None

    # Toc Device
    device__toc__probability:										Optional[float] = None
    device__toc__explicit:											Optional[str] = None
    device__toc__deterministic:										Optional[bool] = None
    device__toc__manualType:										Optional[str] = None
    device__toc__automatedType:										Optional[str] = None
    device__toc__responseTime:										Optional[float] = None
    device__toc__recoveryRate:										Optional[float] = None
    device__toc__lcAbstinence:										Optional[float] = None
    device__toc__initialAwareness:									Optional[float] = None
    device__toc__mrmDecel:											Optional[float] = None
    device__toc__dynamicToCThreshold:								Optional[float] = None
    device__toc__dynamicMRMProbability:								Optional[float] = None
    device__toc__mrmKeepRight:										Optional[bool] = None
    device__toc__mrmSafeSpot:										Optional[str] = None
    device__toc__mrmSafeSpotDuration:								Optional[float] = None
    device__toc__maxPreparationAccel:								Optional[float] = None
    device__toc__ogNewTimeHeadway:									Optional[float] = None
    device__toc__ogNewSpaceHeadway:									Optional[float] = None
    device__toc__ogMaxDecel:										Optional[float] = None
    device__toc__ogChangeRate:										Optional[float] = None
    device__toc__useColorScheme:									Optional[bool] = None
    device__toc__file:												Optional[str] = None

    # Driver State Device
    device__driverstate__probability:								Optional[float] = None
    device__driverstate__explicit:									Optional[str] = None
    device__driverstate__deterministic:								Optional[bool] = None
    device__driverstate__initialAwareness:							Optional[float] = None
    device__driverstate__errorTimeScaleCoefficient:					Optional[float] = None
    device__driverstate__errorNoiseIntensityCoefficient:			Optional[float] = None
    device__driverstate__speedDifferenceErrorCoefficient:			Optional[float] = None
    device__driverstate__headwayErrorCoefficient:					Optional[float] = None
    device__driverstate__speedDifferenceChangePerceptionThreshold:	Optional[float] = None
    device__driverstate__headwayChangePerceptionThreshold:			Optional[float] = None
    device__driverstate__minAwareness:								Optional[float] = None
    device__driverstate__maximalReactionTime:						Optional[float] = None

    # Bluelight Device
    device__bluelight__probability:									Optional[float] = None
    device__bluelight__explicit:									Optional[str] = None
    device__bluelight__deterministic:								Optional[bool] = None
    device__bluelight__reactiondist:								Optional[float] = None

    # Fcd Device
    device__fcd__probability:										Optional[float] = None
    device__fcd__explicit:											Optional[str] = None
    device__fcd__deterministic:										Optional[bool] = None
    device__fcd__begin:												Optional[str] = None
    device__fcd__period:											Optional[str] = None
    device__fcd__radius:											Optional[float] = None
    person_device__fcd__probability:								Optional[float] = None
    person_device__fcd__explicit:									Optional[str] = None
    person_device__fcd__deterministic:								Optional[bool] = None
    person_device__fcd__period:										Optional[str] = None

    # Elechybrid Device
    device__elechybrid__probability:								Optional[float] = None
    device__elechybrid__explicit:									Optional[str] = None
    device__elechybrid__deterministic:								Optional[bool] = None

    # Taxi Device
    device__taxi__probability:										Optional[float] = None
    device__taxi__explicit:											Optional[str] = None
    device__taxi__deterministic:									Optional[bool] = None
    device__taxi__dispatch_algorithm:								Optional[str] = None
    device__taxi__dispatch_algorithm__output:						Optional[str] = None
    device__taxi__dispatch_algorithm__params:						Optional[str] = None
    device__taxi__dispatch_period:									Optional[float] = None
    device__taxi__idle_algorithm:									Optional[str] = None
    device__taxi__idle_algorithm__output:							Optional[str] = None

    # Glosa Device
    device__glosa__probability:										Optional[float] = None
    device__glosa__explicit:										Optional[str] = None
    device__glosa__deterministic:									Optional[bool] = None
    device__glosa__range:											Optional[float] = None
    device__glosa__max_speedfactor:									Optional[float] = None
    device__glosa__min_speed:										Optional[float] = None

    # Tripinfo Device
    device__tripinfo__probability:									Optional[float] = None
    device__tripinfo__explicit:										Optional[str] = None
    device__tripinfo__deterministic:								Optional[bool] = None

    # Vehroutes Device
    device__vehroute__probability:									Optional[float] = None
    device__vehroute__explicit:										Optional[str] = None
    device__vehroute__deterministic:								Optional[bool] = None

    # Friction Device
    device__friction__probability:									Optional[float] = None
    device__friction__explicit:										Optional[str] = None
    device__friction__deterministic:								Optional[bool] = None
    device__friction__stdDev:										Optional[float] = None
    device__friction__offset:										Optional[float] = None

    # Traci Server
    remote_port:													Optional[int] = None
    num_clients:													Optional[int] = None

    # Mesoscopic
    mesosim:														Optional[bool] = None
    meso_edgelength:												Optional[float] = None
    meso_tauff:														Optional[float] = None
    meso_taufj:														Optional[float] = None
    meso_taujf:														Optional[float] = None
    meso_taujj:														Optional[float] = None
    meso_jam_threshold:												Optional[float] = None
    meso_multi_queue:												Optional[bool] = None
    meso_lane_queue:												Optional[bool] = None
    meso_ignore_lanes_by_vclass:									Optional[str] = None
    meso_junction_control:											Optional[bool] = None
    meso_junction_control__limited:									Optional[bool] = None
    meso_tls_penalty:												Optional[float] = None
    meso_tls_flow_penalty:											Optional[float] = None
    meso_minor_penalty:												Optional[float] = None
    meso_overtaking:												Optional[bool] = None
    meso_recheck:													Optional[float] = None

    # Random Number
    random:															Optional[bool] = None
    seed:															Optional[int] = None
    thread_rngs:													Optional[int] = None

    # Gui Only
    gui_settings_file:												Optional[str] = None
    quit_on_end:													Optional[bool] = None
    game:															Optional[bool] = None
    game__mode:														Optional[str] = None
    start:															Optional[bool] = None
    delay:															Optional[float] = None
    breakpoints:													Optional[List[float]] = None
    edgedata_files:													Optional[str] = None
    demo:															Optional[bool] = None
    disable_textures:												Optional[bool] = None
    registry_viewport:												Optional[bool] = None
    window_size:													Optional[List[float]] = None
    window_pos:														Optional[List[float]] = None
    tracker_interval:												Optional[float] = None
    osg_view:														Optional[bool] = None
    gui_testing:													Optional[bool] = None
    gui_testing_debug:												Optional[bool] = None
    gui_testing__setting_output:									Optional[str] = None

    def generate_command(self) -> List[str]:
        """
        takes attribute names which are not None and don't start with underscore
        and generates a list of flag names and corresponding values
        for sumo call
        """
        cmd = [self._sumo_binary]
        for (attr_name, value) in self._get_valid_attr_list():
            cmd.append(attr_name)
            cmd.append(value)
        return cmd

    def _get_valid_attr_list(self) -> List[Tuple[str, str]]:
        """
        returns a list of tuples with flag names and corresponding values
        """
        transform_tuple = lambda tpl: (
                _name_to_flag(tpl[0]),
                _attr_value_to_str(tpl[1])
            )
        return [transform_tuple(tpl) for tpl in inspect.getmembers(self) if _is_valid_flag(tpl)]

    def save_config(self):
        """
        feeds all valid flags to sumo and
        saves config file using --save-configuration flag
        link: https://sumo.dlr.de/docs/Basics/Using_the_Command_Line_Applications.html#generating_configuration_files_templates_and_schemata
        """
        output_folder = f'{os.path.dirname(__file__)}/config_files'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = f'{output_folder}/sumo_saved_config.xml'
        cmd = self.generate_command()
        cmd.extend(['--save-configuration', output_path])
        subprocess.run(cmd)


def _name_to_flag(name: str) -> str:
    return '--' + name.replace('__', '.').replace('_', '-')


def _is_valid_flag(attr_tuple: Tuple[str, Optional[str]]) -> bool:
    (attr_name, value) = attr_tuple
    return not (attr_name.startswith('_') or value is None or inspect.ismethod(value))


def _attr_value_to_str(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)

