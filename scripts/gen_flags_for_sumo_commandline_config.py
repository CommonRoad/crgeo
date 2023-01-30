#just the whole text of the table at https://sumo.dlr.de/docs/sumo.html
text: str = """
Configuration
Option 	Description
-c <FILE>
--configuration-file <FILE> 	Loads the named config on startup
-C <FILE>
--save-configuration <FILE> 	Saves current configuration into FILE
--save-configuration.relative <BOOL> 	Enforce relative paths when saving the configuration; default: false
--save-template <FILE> 	Saves a configuration template (empty) into FILE
--save-schema <FILE> 	Saves the configuration schema into FILE
--save-commented <BOOL> 	Adds comments to saved template, configuration, or schema; default: false
Input
Option 	Description
-n <FILE>
--net-file <FILE> 	Load road network description from FILE
-r <FILE>
--route-files <FILE> 	Load routes descriptions from FILE(s)
-a <FILE>
--additional-files <FILE> 	Load further descriptions from FILE(s)
-w <FILE>
--weight-files <FILE> 	Load edge/lane weights for online rerouting from FILE
-x <STRING>
--weight-attribute <STRING> 	Name of the xml attribute which gives the edge weight; default: traveltime
--load-state <FILE> 	Loads a network state from FILE
--load-state.offset <TIME> 	Shifts all times loaded from a saved state by the given offset; default: 0
--load-state.remove-vehicles 	Removes vehicles with the given IDs from the loaded state
--junction-taz <BOOL> 	Initialize a TAZ for every junction to use attributes toJunction and fromJunction; default: false
Output
Option 	Description
--write-license <BOOL> 	Include license info into every output file; default: false
--output-prefix <STRING> 	Prefix which is applied to all output files. The special string 'TIME' is replaced by the current time.
--precision <INT> 	Defines the number of digits after the comma for floating point output; default: 2
--precision.geo <INT> 	Defines the number of digits after the comma for lon,lat output; default: 6
-H <BOOL>
--human-readable-time <BOOL> 	Write time values as hour:minute:second or day:hour:minute:second rather than seconds; default: false
--netstate-dump <FILE> 	Save complete network states into FILE
--netstate-dump.empty-edges <BOOL> 	Write also empty edges completely when dumping; default: false
--netstate-dump.precision <INT> 	Write positions and speeds with the given precision (default 2); default: 2
--emission-output <FILE> 	Save the emission values of each vehicle
--emission-output.precision <INT> 	Write emission values with the given precision (default 2); default: 2
--emission-output.geo <BOOL> 	Save the positions in emission output using geo-coordinates (lon/lat); default: false
--emission-output.step-scaled <BOOL> 	Write emission values scaled to the step length rather than as per-second values; default: false
--battery-output <FILE> 	Save the battery values of each vehicle
--battery-output.precision <INT> 	Write battery values with the given precision (default 2); default: 2
--elechybrid-output <FILE> 	Save the elecHybrid values of each vehicle
--elechybrid-output.precision <INT> 	Write elecHybrid values with the given precision (default 2); default: 2
--elechybrid-output.aggregated <BOOL> 	Write elecHybrid values into one aggregated file; default: false
--chargingstations-output <FILE> 	Write data of charging stations
--overheadwiresegments-output <FILE> 	Write data of overhead wire segments
--substations-output <FILE> 	Write data of electrical substation stations
--substations-output.precision <INT> 	Write substation values with the given precision (default 2); default: 2
--fcd-output <FILE> 	Save the Floating Car Data
--fcd-output.geo <BOOL> 	Save the Floating Car Data using geo-coordinates (lon/lat); default: false
--fcd-output.signals <BOOL> 	Add the vehicle signal state to the FCD output (brake lights etc.); default: false
--fcd-output.distance <BOOL> 	Add kilometrage to the FCD output (linear referencing); default: false
--fcd-output.acceleration <BOOL> 	Add acceleration to the FCD output; default: false
--fcd-output.max-leader-distance <FLOAT> 	Add leader vehicle information to the FCD output (within the given distance); default: -1
--fcd-output.params 	Add generic parameter values to the FCD output
--fcd-output.filter-edges.input-file <FILE> 	Restrict fcd output to the edge selection from the given input file
--fcd-output.attributes 	List attributes that should be included in the FCD output
--fcd-output.filter-shapes 	List shape names that should be used to filter the FCD output
--device.ssm.filter-edges.input-file <FILE> 	Restrict SSM device output to the edge selection from the given input file
--full-output <FILE> 	Save a lot of information for each timestep (very redundant)
--queue-output <FILE> 	Save the vehicle queues at the junctions (experimental)
--queue-output.period <TIME> 	Save vehicle queues with the given period; default: -1
--vtk-output <FILE> 	Save complete vehicle positions inclusive speed values in the VTK Format (usage: /path/out will produce /path/out_$TIMESTEP$.vtp files)
--amitran-output <FILE> 	Save the vehicle trajectories in the Amitran format
--summary-output <FILE> 	Save aggregated vehicle departure info into FILE
--summary-output.period <TIME> 	Save summary-output with the given period; default: -1
--person-summary-output <FILE> 	Save aggregated person counts into FILE
--tripinfo-output <FILE> 	Save single vehicle trip info into FILE
--tripinfo-output.write-unfinished <BOOL> 	Write tripinfo output for vehicles which have not arrived at simulation end; default: false
--tripinfo-output.write-undeparted <BOOL> 	Write tripinfo output for vehicles which have not departed at simulation end because of depart delay; default: false
--vehroute-output <FILE> 	Save single vehicle route info into FILE
--vehroute-output.exit-times <BOOL> 	Write the exit times for all edges; default: false
--vehroute-output.last-route <BOOL> 	Write the last route only; default: false
--vehroute-output.sorted <BOOL> 	Sorts the output by departure time; default: false
--vehroute-output.dua <BOOL> 	Write the output in the duarouter alternatives style; default: false
--vehroute-output.cost <BOOL> 	Write costs for all routes; default: false
--vehroute-output.intended-depart <BOOL> 	Write the output with the intended instead of the real departure time; default: false
--vehroute-output.route-length <BOOL> 	Include total route length in the output; default: false
--vehroute-output.write-unfinished <BOOL> 	Write vehroute output for vehicles which have not arrived at simulation end; default: false
--vehroute-output.skip-ptlines <BOOL> 	Skip vehroute output for public transport vehicles; default: false
--vehroute-output.incomplete <BOOL> 	Include invalid routes and route stubs in vehroute output; default: false
--vehroute-output.stop-edges <BOOL> 	Include information about edges between stops; default: false
--vehroute-output.speedfactor <BOOL> 	Write the vehicle speedFactor (defaults to 'true' if departSpeed is written); default: false
--vehroute-output.internal <BOOL> 	Include internal edges in the output; default: false
--personroute-output <FILE> 	Save person and container routes to separate FILE
--link-output <FILE> 	Save links states into FILE
--railsignal-block-output <FILE> 	Save railsignal-blocks into FILE
--bt-output <FILE> 	Save bluetooth visibilities into FILE (in conjunction with device.btreceiver and device.btsender)
--lanechange-output <FILE> 	Record lane changes and their motivations for all vehicles into FILE
--lanechange-output.started <BOOL> 	Record start of lane change manoeuvres; default: false
--lanechange-output.ended <BOOL> 	Record end of lane change manoeuvres; default: false
--lanechange-output.xy <BOOL> 	Record coordinates of lane change manoeuvres; default: false
--stop-output <FILE> 	Record stops and loading/unloading of passenger and containers for all vehicles into FILE
--stop-output.write-unfinished <BOOL> 	Write stop output for stops which have not ended at simulation end; default: false
--collision-output <FILE> 	Write collision information into FILE
--edgedata-output <FILE> 	Write aggregated traffic statistics for all edges into FILE
--lanedata-output <FILE> 	Write aggregated traffic statistics for all lanes into FILE
--statistic-output <FILE> 	Write overall statistics into FILE
--save-state.times 	Use TIME[] as times at which a network state written
--save-state.period <TIME> 	save state repeatedly after TIME period; default: -1
--save-state.period.keep <INT> 	Keep only the last INT periodic state files; default: 0
--save-state.prefix <FILE> 	Prefix for network states; default: state
--save-state.suffix <STRING> 	Suffix for network states (.xml.gz or .xml); default: .xml.gz
--save-state.files <FILE> 	Files for network states
--save-state.rng <BOOL> 	Save random number generator states; default: false
--save-state.transportables <BOOL> 	Save person and container states (experimental); default: false
--save-state.constraints <BOOL> 	Save rail signal constraints; default: false
--save-state.precision <INT> 	Write internal state values with the given precision (default 2); default: 2
Time
Option 	Description
-b <TIME>
--begin <TIME> 	Defines the begin time in seconds; The simulation starts at this time; default: 0
-e <TIME>
--end <TIME> 	Defines the end time in seconds; The simulation ends at this time; default: -1
--step-length <TIME> 	Defines the step duration in seconds; default: 1
Processing
Option 	Description
--step-method.ballistic <BOOL> 	Whether to use ballistic method for the positional update of vehicles (default is a semi-implicit Euler method).; default: false
--extrapolate-departpos <BOOL> 	Whether vehicles that depart between simulation steps should extrapolate the depart position; default: false
--threads <INT> 	Defines the number of threads for parallel simulation; default: 1
--lateral-resolution <FLOAT> 	Defines the resolution in m when handling lateral positioning within a lane (with -1 all vehicles drive at the center of their lane; default: -1
-s <TIME>
--route-steps <TIME> 	Load routes for the next number of seconds ahead; default: 200
--no-internal-links <BOOL> 	Disable (junction) internal links; default: false
--ignore-junction-blocker <TIME> 	Ignore vehicles which block the junction after they have been standing for SECONDS (-1 means never ignore); default: -1
--ignore-route-errors <BOOL> 	Do not check whether routes are connected; default: false
--ignore-accidents <BOOL> 	Do not check whether accidents occur; default: false
--collision.action <STRING> 	How to deal with collisions: [none,warn,teleport,remove]; default: teleport
--collision.stoptime <TIME> 	Let vehicle stop for TIME before performing collision.action (except for action 'none'); default: 0
--collision.check-junctions <BOOL> 	Enables collisions checks on junctions; default: false
--collision.check-junctions.mingap <FLOAT> 	Increase or decrease sensitivity for junction collision check; default: 0
--collision.mingap-factor <FLOAT> 	Sets the fraction of minGap that must be maintained to avoid collision detection. If a negative value is given, the carFollowModel parameter is used; default: -1
--max-num-vehicles <INT> 	Delay vehicle insertion to stay within the given maximum number; default: -1
--max-num-teleports <INT> 	Abort the simulation if the given maximum number of teleports is exceeded; default: -1
--scale <FLOAT> 	Scale demand by the given factor (by discarding or duplicating vehicles); default: 1
--scale-suffix <STRING> 	Suffix to be added when creating ids for cloned vehicles; default: .
--time-to-teleport <TIME> 	Specify how long a vehicle may wait until being teleported, defaults to 300, non-positive values disable teleporting; default: 300
--time-to-teleport.highways <TIME> 	The waiting time after which vehicles on a fast road (speed > 69km/h) are teleported if they are on a non-continuing lane; default: 0
--time-to-teleport.highways.min-speed <FLOAT> 	The waiting time after which vehicles on a fast road (default: speed > 69km/h) are teleported if they are on a non-continuing lane; default: 19.1667
--time-to-teleport.disconnected <TIME> 	The waiting time after which vehicles with a disconnected route are teleported. Negative values disable teleporting; default: -1
--time-to-teleport.remove <BOOL> 	Whether vehicles shall be removed after waiting too long instead of being teleported; default: false
--time-to-teleport.ride <TIME> 	The waiting time after which persons / containers waiting for a pickup are teleported. Negative values disable teleporting; default: -1
--time-to-teleport.bidi <TIME> 	The waiting time after which vehicles on bidirectional edges are teleported; default: -1
--waiting-time-memory <TIME> 	Length of time interval, over which accumulated waiting time is taken into account (default is 100s.); default: 100
--startup-wait-threshold <TIME> 	Minimum consecutive waiting time before applying startupDelay; default: 2
--max-depart-delay <TIME> 	How long vehicles wait for departure before being skipped, defaults to -1 which means vehicles are never skipped; default: -1
--sloppy-insert <BOOL> 	Whether insertion on an edge shall not be repeated in same step once failed; default: false
--eager-insert <BOOL> 	Whether each vehicle is checked separately for insertion on an edge; default: false
--emergency-insert <BOOL> 	Allow inserting a vehicle in a situation which requires emergency braking; default: false
--random-depart-offset <TIME> 	Each vehicle receives a random offset to its depart value drawn uniformly from [0, TIME]; default: 0
--lanechange.duration <TIME> 	Duration of a lane change maneuver (default 0); default: 0
--lanechange.overtake-right <BOOL> 	Whether overtaking on the right on motorways is permitted; default: false
--tls.all-off <BOOL> 	Switches off all traffic lights.; default: false
--tls.actuated.show-detectors <BOOL> 	Sets default visibility for actuation detectors; default: false
--tls.actuated.jam-threshold <FLOAT> 	Sets default jam-treshold parameter for all actuation detectors; default: -1
--tls.actuated.detector-length <FLOAT> 	Sets default detector length parameter for all actuation detectors; default: 0
--tls.delay_based.detector-range <FLOAT> 	Sets default range for detecting delayed vehicles; default: 100
--tls.yellow.min-decel <FLOAT> 	Minimum deceleration when braking at yellow; default: 3
--railsignal-moving-block <BOOL> 	Let railsignals operate in moving-block mode by default; default: false
--time-to-impatience <TIME> 	Specify how long a vehicle may wait until impatience grows from 0 to 1, defaults to 300, non-positive values disable impatience growth; default: 180
--default.action-step-length <FLOAT> 	Length of the default interval length between action points for the car-following and lane-change models (in seconds). If not specified, the simulation step-length is used per default. Vehicle- or VType-specific settings override the default. Must be a multiple of the simulation step-length.; default: 0
--default.carfollowmodel <STRING> 	Select default car following model (Krauss, IDM, ...); default: Krauss
--default.speeddev <FLOAT> 	Select default speed deviation. A negative value implies vClass specific defaults (0.1 for the default passenger class; default: -1
--default.emergencydecel <STRING> 	Select default emergencyDecel value among ('decel', 'default', FLOAT) which sets the value either to the same as the deceleration value, a vClass-class specific default or the given FLOAT in m/s^2; default: default
--overhead-wire.solver <BOOL> 	Use Kirchhoff's laws for solving overhead wire circuit; default: true
--overhead-wire.recuperation <BOOL> 	Enable recuperation from the vehicle equipped with elecHybrid device into the ovrehead wire.; default: true
--overhead-wire.substation-current-limits <BOOL> 	Enable current limits of traction substation during solving the overhead wire electrical circuit.; default: true
--emergencydecel.warning-threshold <FLOAT> 	Sets the fraction of emergency decel capability that must be used to trigger a warning.; default: 1
--parking.maneuver <BOOL> 	Whether parking simulation includes manoeuvering time and associated lane blocking; default: false
--use-stop-ended <BOOL> 	Override stop until times with stop ended times when given; default: false
--pedestrian.model <STRING> 	Select among pedestrian models ['nonInteracting', 'striping', 'remote']; default: striping
--pedestrian.striping.stripe-width <FLOAT> 	Width of parallel stripes for segmenting a sidewalk (meters) for use with model 'striping'; default: 0.64
--pedestrian.striping.dawdling <FLOAT> 	Factor for random slow-downs [0,1] for use with model 'striping'; default: 0.2
--pedestrian.striping.mingap-to-vehicle <FLOAT> 	Minimal gap / safety buffer (in meters) from a pedestrian to another vehicle for use with model 'striping'; default: 0.25
--pedestrian.striping.jamtime <TIME> 	Time in seconds after which pedestrians start squeezing through a jam when using model 'striping' (non-positive values disable squeezing); default: 300
--pedestrian.striping.jamtime.crossing <TIME> 	Time in seconds after which pedestrians start squeezing through a jam while on a pedestrian crossing when using model 'striping' (non-positive values disable squeezing); default: 10
--pedestrian.striping.jamtime.narrow <TIME> 	Time in seconds after which pedestrians start squeezing through a jam while on a narrow lane when using model 'striping'; default: 1
--pedestrian.striping.reserve-oncoming <FLOAT> 	Fraction of stripes to reserve for oncoming pedestrians; default: 0
--pedestrian.striping.reserve-oncoming.junctions <FLOAT> 	Fraction of stripes to reserve for oncoming pedestrians on crossings and walkingareas; default: 0.34
--pedestrian.striping.legacy-departposlat <BOOL> 	Interpret departPosLat for walks in legacy style; default: false
--pedestrian.striping.walkingarea-detail <INT> 	Generate INT intermediate points to smooth out lanes within the walkingarea; default: 4
--pedestrian.remote.address <STRING> 	The address (host:port) of the external simulation; default: localhost:9000
--ride.stop-tolerance <FLOAT> 	Tolerance to apply when matching pedestrian and vehicle positions on boarding at individual stops; default: 10
--persontrip.walk-opposite-factor <FLOAT> 	Use FLOAT as a factor on walking speed against vehicle traffic direction; default: 1
Routing
Option 	Description
--routing-algorithm <STRING> 	Select among routing algorithms ['dijkstra', 'astar', 'CH', 'CHWrapper']; default: dijkstra
--weights.random-factor <FLOAT> 	Edge weights for routing are dynamically disturbed by a random factor drawn uniformly from [1,FLOAT); default: 1
--weights.minor-penalty <FLOAT> 	Apply the given time penalty when computing minimum routing costs for minor-link internal lanes; default: 1.5
--weights.tls-penalty <FLOAT> 	Apply scaled travel time penalties based on green split when computing minimum routing costs for internal lanes at traffic lights; default: 0
--weights.priority-factor <FLOAT> 	Consider edge priorities in addition to travel times, weighted by factor; default: 0
--weights.separate-turns <FLOAT> 	Distinguish travel time by turn direction and shift a fraction of the estimated time loss ahead of the intersection onto the internal edges; default: 0
--astar.all-distances <FILE> 	Initialize lookup table for astar from the given file (generated by marouter --all-pairs-output)
--astar.landmark-distances <FILE> 	Initialize lookup table for astar ALT-variant from the given file
--persontrip.walkfactor <FLOAT> 	Use FLOAT as a factor on pedestrian maximum speed during intermodal routing; default: 0.75
--persontrip.transfer.car-walk 	Where are mode changes from car to walking allowed (possible values: 'parkingAreas', 'ptStops', 'allJunctions' and combinations); default: parkingAreas
--persontrip.transfer.taxi-walk 	Where taxis can drop off customers ('allJunctions, 'ptStops')
--persontrip.transfer.walk-taxi 	Where taxis can pick up customers ('allJunctions, 'ptStops')
--persontrip.default.group <STRING> 	When set, trips between the same origin and destination will share a taxi by default
--persontrip.taxi.waiting-time <TIME> 	Estimated time for taxi pickup; default: 300
--railway.max-train-length <FLOAT> 	Use FLOAT as a maximum train length when initializing the railway router; default: 1000
--replay-rerouting <BOOL> 	Replay exact rerouting sequence from vehroute-output; default: false
--device.rerouting.probability <FLOAT> 	The probability for a vehicle to have a 'rerouting' device; default: -1
--device.rerouting.explicit 	Assign a 'rerouting' device to named vehicles
--device.rerouting.deterministic <BOOL> 	The 'rerouting' devices are set deterministic using a fraction of 1000; default: false
--device.rerouting.period <TIME> 	The period with which the vehicle shall be rerouted; default: 0
--device.rerouting.pre-period <TIME> 	The rerouting period before depart; default: 60
--device.rerouting.adaptation-weight <FLOAT> 	The weight of prior edge weights for exponential moving average; default: 0
--device.rerouting.adaptation-steps <INT> 	The number of steps for moving average weight of prior edge weights; default: 180
--device.rerouting.adaptation-interval <TIME> 	The interval for updating the edge weights; default: 1
--device.rerouting.with-taz <BOOL> 	Use zones (districts) as routing start- and endpoints; default: false
--device.rerouting.init-with-loaded-weights <BOOL> 	Use weight files given with option --weight-files for initializing edge weights; default: false
--device.rerouting.threads <INT> 	The number of parallel execution threads used for rerouting; default: 0
--device.rerouting.synchronize <BOOL> 	Let rerouting happen at the same time for all vehicles; default: false
--device.rerouting.railsignal <BOOL> 	Allow rerouting triggered by rail signals.; default: true
--device.rerouting.bike-speeds <BOOL> 	Compute separate average speeds for bicycles; default: false
--device.rerouting.output <FILE> 	Save adapting weights to FILE
--person-device.rerouting.probability <FLOAT> 	The probability for a person to have a 'rerouting' device; default: -1
--person-device.rerouting.explicit 	Assign a 'rerouting' device to named persons
--person-device.rerouting.deterministic <BOOL> 	The 'rerouting' devices are set deterministic using a fraction of 1000; default: false
--person-device.rerouting.period <TIME> 	The period with which the person shall be rerouted; default: 0
Report
Option 	Description
-v <BOOL>
--verbose <BOOL> 	Switches to verbose output; default: false
--print-options <BOOL> 	Prints option values before processing; default: false
-? <BOOL>
--help <BOOL> 	Prints this screen or selected topics; default: false
-V <BOOL>
--version <BOOL> 	Prints the current version; default: false
-X <STRING>
--xml-validation <STRING> 	Set schema validation scheme of XML inputs ("never", "local", "auto" or "always"); default: local
--xml-validation.net <STRING> 	Set schema validation scheme of SUMO network inputs ("never", "local", "auto" or "always"); default: never
--xml-validation.routes <STRING> 	Set schema validation scheme of SUMO route inputs ("never", "local", "auto" or "always"); default: local
-W <BOOL>
--no-warnings <BOOL> 	Disables output of warnings; default: false
--aggregate-warnings <INT> 	Aggregate warnings of the same type whenever more than INT occur; default: -1
-l <FILE>
--log <FILE> 	Writes all messages to FILE (implies verbose)
--message-log <FILE> 	Writes all non-error messages to FILE (implies verbose)
--error-log <FILE> 	Writes all warnings and errors to FILE
--duration-log.disable <BOOL> 	Disable performance reports for individual simulation steps; default: false
-t <BOOL>
--duration-log.statistics <BOOL> 	Enable statistics on vehicle trips; default: false
--no-step-log <BOOL> 	Disable console output of current simulation step; default: false
--step-log.period <INT> 	Number of simulation steps between step-log outputs; default: 100
Emissions
Option 	Description
--emissions.volumetric-fuel <BOOL> 	Return fuel consumption values in (legacy) unit l instead of mg; default: false
--phemlight-path <FILE> 	Determines where to load PHEMlight definitions from; default: ./PHEMlight/
--phemlight-year <INT> 	Enable fleet age modelling with the given reference year in PHEMlight5; default: 0
--phemlight-temperature <FLOAT> 	Set ambient temperature to correct NOx emissions in PHEMlight5; default: 1.79769e+308
--device.emissions.probability <FLOAT> 	The probability for a vehicle to have a 'emissions' device; default: -1
--device.emissions.explicit 	Assign a 'emissions' device to named vehicles
--device.emissions.deterministic <BOOL> 	The 'emissions' devices are set deterministic using a fraction of 1000; default: false
--device.emissions.begin <STRING> 	Recording begin time for emission-data; default: -1
--device.emissions.period <STRING> 	Recording period for emission-output; default: 0
Communication
Option 	Description
--device.btreceiver.probability <FLOAT> 	The probability for a vehicle to have a 'btreceiver' device; default: -1
--device.btreceiver.explicit 	Assign a 'btreceiver' device to named vehicles
--device.btreceiver.deterministic <BOOL> 	The 'btreceiver' devices are set deterministic using a fraction of 1000; default: false
--device.btreceiver.range <FLOAT> 	The range of the bt receiver; default: 300
--device.btreceiver.all-recognitions <BOOL> 	Whether all recognition point shall be written; default: false
--device.btreceiver.offtime <FLOAT> 	The offtime used for calculating detection probability (in seconds); default: 0.64
--device.btsender.probability <FLOAT> 	The probability for a vehicle to have a 'btsender' device; default: -1
--device.btsender.explicit 	Assign a 'btsender' device to named vehicles
--device.btsender.deterministic <BOOL> 	The 'btsender' devices are set deterministic using a fraction of 1000; default: false
--person-device.btsender.probability <FLOAT> 	The probability for a person to have a 'btsender' device; default: -1
--person-device.btsender.explicit 	Assign a 'btsender' device to named persons
--person-device.btsender.deterministic <BOOL> 	The 'btsender' devices are set deterministic using a fraction of 1000; default: false
--person-device.btreceiver.probability <FLOAT> 	The probability for a person to have a 'btreceiver' device; default: -1
--person-device.btreceiver.explicit 	Assign a 'btreceiver' device to named persons
--person-device.btreceiver.deterministic <BOOL> 	The 'btreceiver' devices are set deterministic using a fraction of 1000; default: false
Battery
Option 	Description
--device.battery.probability <FLOAT> 	The probability for a vehicle to have a 'battery' device; default: -1
--device.battery.explicit 	Assign a 'battery' device to named vehicles
--device.battery.deterministic <BOOL> 	The 'battery' devices are set deterministic using a fraction of 1000; default: false
--device.battery.track-fuel <BOOL> 	Track fuel consumption for non-electric vehicles; default: false
Example Device
Option 	Description
--device.example.probability <FLOAT> 	The probability for a vehicle to have a 'example' device; default: -1
--device.example.explicit 	Assign a 'example' device to named vehicles
--device.example.deterministic <BOOL> 	The 'example' devices are set deterministic using a fraction of 1000; default: false
--device.example.parameter <FLOAT> 	An exemplary parameter which can be used by all instances of the example device; default: 0
Ssm Device
Option 	Description
--device.ssm.probability <FLOAT> 	The probability for a vehicle to have a 'ssm' device; default: -1
--device.ssm.explicit 	Assign a 'ssm' device to named vehicles
--device.ssm.deterministic <BOOL> 	The 'ssm' devices are set deterministic using a fraction of 1000; default: false
--device.ssm.measures <STRING> 	Specifies which measures will be logged (as a space or comma-separated sequence of IDs in ('TTC', 'DRAC', 'PET'))
--device.ssm.thresholds <STRING> 	Specifies space or comma-separated thresholds corresponding to the specified measures (see documentation and watch the order!). Only events exceeding the thresholds will be logged.
--device.ssm.trajectories <BOOL> 	Specifies whether trajectories will be logged (if false, only the extremal values and times are reported).; default: false
--device.ssm.range <FLOAT> 	Specifies the detection range in meters. For vehicles below this distance from the equipped vehicle, SSM values are traced.; default: 50
--device.ssm.extratime <FLOAT> 	Specifies the time in seconds to be logged after a conflict is over. Required >0 if PET is to be calculated for crossing conflicts.; default: 5
--device.ssm.file <STRING> 	Give a global default filename for the SSM output
--device.ssm.geo <BOOL> 	Whether to use coordinates of the original reference system in output; default: false
--device.ssm.write-positions <BOOL> 	Whether to write positions (coordinates) for each timestep; default: false
--device.ssm.write-lane-positions <BOOL> 	Whether to write lanes and their positions for each timestep; default: false
Toc Device
Option 	Description
--device.toc.probability <FLOAT> 	The probability for a vehicle to have a 'toc' device; default: -1
--device.toc.explicit 	Assign a 'toc' device to named vehicles
--device.toc.deterministic <BOOL> 	The 'toc' devices are set deterministic using a fraction of 1000; default: false
--device.toc.manualType <STRING> 	Vehicle type for manual driving regime.
--device.toc.automatedType <STRING> 	Vehicle type for automated driving regime.
--device.toc.responseTime <FLOAT> 	Average response time needed by a driver to take back control.; default: -1
--device.toc.recoveryRate <FLOAT> 	Recovery rate for the driver's awareness after a ToC.; default: 0.1
--device.toc.lcAbstinence <FLOAT> 	Attention level below which a driver restrains from performing lane changes (value in [0,1]).; default: 0
--device.toc.initialAwareness <FLOAT> 	Average awareness a driver has initially after a ToC (value in [0,1]).; default: 0.5
--device.toc.mrmDecel <FLOAT> 	Deceleration rate applied during a 'minimum risk maneuver'.; default: 1.5
--device.toc.dynamicToCThreshold <FLOAT> 	Time, which the vehicle requires to have ahead to continue in automated mode. The default value of 0 indicates no dynamic triggering of ToCs.; default: 0
--device.toc.dynamicMRMProbability <FLOAT> 	Probability that a dynamically triggered TOR is not answered in time.; default: 0.05
--device.toc.mrmKeepRight <BOOL> 	If true, the vehicle tries to change to the right during an MRM.; default: false
--device.toc.mrmSafeSpot <STRING> 	If set, the vehicle tries to reach the given named stopping place during an MRM.
--device.toc.mrmSafeSpotDuration <FLOAT> 	Duration the vehicle stays at the safe spot after an MRM.; default: 60
--device.toc.maxPreparationAccel <FLOAT> 	Maximal acceleration that may be applied during the ToC preparation phase.; default: 0
--device.toc.ogNewTimeHeadway <FLOAT> 	Timegap for ToC preparation phase.; default: -1
--device.toc.ogNewSpaceHeadway <FLOAT> 	Additional spacing for ToC preparation phase.; default: -1
--device.toc.ogMaxDecel <FLOAT> 	Maximal deceleration applied for establishing increased gap in ToC preparation phase.; default: -1
--device.toc.ogChangeRate <FLOAT> 	Rate of adaptation towards the increased headway during ToC preparation.; default: -1
--device.toc.useColorScheme <BOOL> 	Whether a coloring scheme shall by applied to indicate the different ToC stages.; default: true
--device.toc.file <STRING> 	Switches on output by specifying an output filename.
Driver State Device
Option 	Description
--device.driverstate.probability <FLOAT> 	The probability for a vehicle to have a 'driverstate' device; default: -1
--device.driverstate.explicit 	Assign a 'driverstate' device to named vehicles
--device.driverstate.deterministic <BOOL> 	The 'driverstate' devices are set deterministic using a fraction of 1000; default: false
--device.driverstate.initialAwareness <FLOAT> 	Initial value assigned to the driver's awareness.; default: 1
--device.driverstate.errorTimeScaleCoefficient <FLOAT> 	Time scale for the error process.; default: 100
--device.driverstate.errorNoiseIntensityCoefficient <FLOAT> 	Noise intensity driving the error process.; default: 0.2
--device.driverstate.speedDifferenceErrorCoefficient <FLOAT> 	General scaling coefficient for applying the error to the perceived speed difference (error also scales with distance).; default: 0.15
--device.driverstate.headwayErrorCoefficient <FLOAT> 	General scaling coefficient for applying the error to the perceived distance (error also scales with distance).; default: 0.75
--device.driverstate.speedDifferenceChangePerceptionThreshold <FLOAT> 	Base threshold for recognizing changes in the speed difference (threshold also scales with distance).; default: 0.1
--device.driverstate.headwayChangePerceptionThreshold <FLOAT> 	Base threshold for recognizing changes in the headway (threshold also scales with distance).; default: 0.1
--device.driverstate.minAwareness <FLOAT> 	Minimal admissible value for the driver's awareness.; default: 0.1
--device.driverstate.maximalReactionTime <FLOAT> 	Maximal reaction time (~action step length) induced by decreased awareness level (reached for awareness=minAwareness).; default: -1
Bluelight Device
Option 	Description
--device.bluelight.probability <FLOAT> 	The probability for a vehicle to have a 'bluelight' device; default: -1
--device.bluelight.explicit 	Assign a 'bluelight' device to named vehicles
--device.bluelight.deterministic <BOOL> 	The 'bluelight' devices are set deterministic using a fraction of 1000; default: false
--device.bluelight.reactiondist <FLOAT> 	Set the distance at which other drivers react to the blue light and siren sound; default: 25
Fcd Device
Option 	Description
--device.fcd.probability <FLOAT> 	The probability for a vehicle to have a 'fcd' device; default: -1
--device.fcd.explicit 	Assign a 'fcd' device to named vehicles
--device.fcd.deterministic <BOOL> 	The 'fcd' devices are set deterministic using a fraction of 1000; default: false
--device.fcd.begin <STRING> 	Recording begin time for FCD-data; default: -1
--device.fcd.period <STRING> 	Recording period for FCD-data; default: 0
--device.fcd.radius <FLOAT> 	Record objects in a radius around equipped vehicles; default: 0
--person-device.fcd.probability <FLOAT> 	The probability for a person to have a 'fcd' device; default: -1
--person-device.fcd.explicit 	Assign a 'fcd' device to named persons
--person-device.fcd.deterministic <BOOL> 	The 'fcd' devices are set deterministic using a fraction of 1000; default: false
--person-device.fcd.period <STRING> 	Recording period for FCD-data; default: 0
Elechybrid Device
Option 	Description
--device.elechybrid.probability <FLOAT> 	The probability for a vehicle to have a 'elechybrid' device; default: -1
--device.elechybrid.explicit 	Assign a 'elechybrid' device to named vehicles
--device.elechybrid.deterministic <BOOL> 	The 'elechybrid' devices are set deterministic using a fraction of 1000; default: false
Taxi Device
Option 	Description
--device.taxi.probability <FLOAT> 	The probability for a vehicle to have a 'taxi' device; default: -1
--device.taxi.explicit 	Assign a 'taxi' device to named vehicles
--device.taxi.deterministic <BOOL> 	The 'taxi' devices are set deterministic using a fraction of 1000; default: false
--device.taxi.dispatch-algorithm <STRING> 	The dispatch algorithm [greedy,greedyClosest,greedyShared,routeExtension,traci]; default: greedy
--device.taxi.dispatch-algorithm.output <FILE> 	Write information from the dispatch algorithm to FILE
--device.taxi.dispatch-algorithm.params <STRING> 	Load dispatch algorithm parameters in format KEY1:VALUE1[,KEY2:VALUE]
--device.taxi.dispatch-period <TIME> 	The period between successive calls to the dispatcher; default: 60
--device.taxi.idle-algorithm <STRING> 	The behavior of idle taxis [stop,randomCircling]; default: stop
--device.taxi.idle-algorithm.output <FILE> 	Write information from the idling algorithm to FILE
Glosa Device
Option 	Description
--device.glosa.probability <FLOAT> 	The probability for a vehicle to have a 'glosa' device; default: -1
--device.glosa.explicit 	Assign a 'glosa' device to named vehicles
--device.glosa.deterministic <BOOL> 	The 'glosa' devices are set deterministic using a fraction of 1000; default: false
--device.glosa.range <FLOAT> 	The communication range to the traffic light; default: 100
--device.glosa.max-speedfactor <FLOAT> 	The maximum speed factor when approaching a green light; default: 1.1
--device.glosa.min-speed <FLOAT> 	Minimum speed when coasting towards a red light; default: 5
Tripinfo Device
Option 	Description
--device.tripinfo.probability <FLOAT> 	The probability for a vehicle to have a 'tripinfo' device; default: -1
--device.tripinfo.explicit 	Assign a 'tripinfo' device to named vehicles
--device.tripinfo.deterministic <BOOL> 	The 'tripinfo' devices are set deterministic using a fraction of 1000; default: false
Vehroutes Device
Option 	Description
--device.vehroute.probability <FLOAT> 	The probability for a vehicle to have a 'vehroute' device; default: -1
--device.vehroute.explicit 	Assign a 'vehroute' device to named vehicles
--device.vehroute.deterministic <BOOL> 	The 'vehroute' devices are set deterministic using a fraction of 1000; default: false
Friction Device
Option 	Description
--device.friction.probability <FLOAT> 	The probability for a vehicle to have a 'friction' device; default: -1
--device.friction.explicit 	Assign a 'friction' device to named vehicles
--device.friction.deterministic <BOOL> 	The 'friction' devices are set deterministic using a fraction of 1000; default: false
--device.friction.stdDev <FLOAT> 	The measurement noise parameter which can be applied to the friction device; default: 0.1
--device.friction.offset <FLOAT> 	The measurement offset parameter which can be applied to the friction device -> e.g. to force false measurements; default: 0
Traci Server
Option 	Description
--remote-port <INT> 	Enables TraCI Server if set; default: 0
--num-clients <INT> 	Expected number of connecting clients; default: 1
Mesoscopic
Option 	Description
--mesosim <BOOL> 	Enables mesoscopic simulation; default: false
--meso-edgelength <FLOAT> 	Length of an edge segment in mesoscopic simulation; default: 98
--meso-tauff <TIME> 	Factor for calculating the net free-free headway time; default: 1.13
--meso-taufj <TIME> 	Factor for calculating the net free-jam headway time; default: 1.13
--meso-taujf <TIME> 	Factor for calculating the jam-free headway time; default: 1.73
--meso-taujj <TIME> 	Factor for calculating the jam-jam headway time; default: 1.4
--meso-jam-threshold <FLOAT> 	Minimum percentage of occupied space to consider a segment jammed. A negative argument causes thresholds to be computed based on edge speed and tauff (default); default: -1
--meso-multi-queue <BOOL> 	Enable multiple queues at edge ends; default: true
--meso-lane-queue <BOOL> 	Enable separate queues for every lane; default: false
--meso-ignore-lanes-by-vclass 	Do not build queues (or reduce capacity) for lanes allowing only the given vclasses; default: pedestrian,bicycle
--meso-junction-control <BOOL> 	Enable mesoscopic traffic light and priority junction handling; default: false
--meso-junction-control.limited <BOOL> 	Enable mesoscopic traffic light and priority junction handling for saturated links. This prevents faulty traffic lights from hindering flow in low-traffic situations; default: false
--meso-tls-penalty <FLOAT> 	Apply scaled travel time penalties when driving across tls controlled junctions based on green split instead of checking actual phases; default: 0
--meso-tls-flow-penalty <FLOAT> 	Apply scaled headway penalties when driving across tls controlled junctions based on green split instead of checking actual phases; default: 0
--meso-minor-penalty <TIME> 	Apply fixed time penalty when driving across a minor link. When using --meso-junction-control.limited, the penalty is not applied whenever limited control is active.; default: 0
--meso-overtaking <BOOL> 	Enable mesoscopic overtaking; default: false
--meso-recheck <TIME> 	Time interval for rechecking insertion into the next segment after failure; default: 0
Random Number
Option 	Description
--random <BOOL> 	Initialises the random number generator with the current system time; default: false
--seed <INT> 	Initialises the random number generator with the given value; default: 23423
--thread-rngs <INT> 	Number of pre-allocated random number generators to ensure repeatable multi-threaded simulations (should be at least the number of threads for repeatable simulations).; default: 64
Gui Only
Option 	Description
-g <FILE>
--gui-settings-file <FILE> 	Load visualisation settings from FILE
-Q <BOOL>
--quit-on-end <BOOL> 	Quits the GUI when the simulation stops; default: false
-G <BOOL>
--game <BOOL> 	Start the GUI in gaming mode; default: false
--game.mode <STRING> 	Select the game type ('tls', 'drt'); default: tls
-S <BOOL>
--start <BOOL> 	Start the simulation after loading; default: false
-d <FLOAT>
--delay <FLOAT> 	Use FLOAT in ms as delay between simulation steps; default: 0
-B
--breakpoints 	Use TIME[] as times when the simulation should halt
--edgedata-files <FILE> 	Load edge/lane weights for visualization from FILE
-D <BOOL>
--demo <BOOL> 	Restart the simulation after ending (demo mode); default: false
-T <BOOL>
--disable-textures <BOOL> 	Do not load background pictures; default: false
--registry-viewport <BOOL> 	Load current viewport from registry; default: false
--window-size 	Create initial window with the given x,y size
--window-pos 	Create initial window at the given x,y position
--tracker-interval <TIME> 	The aggregation period for value tracker windows; default: 1
--osg-view <BOOL> 	Start with an OpenSceneGraph view instead of the regular 2D view; default: false
--gui-testing <BOOL> 	Enable overlay for screen recognition; default: false
--gui-testing-debug <BOOL> 	Enable output messages during GUI-Testing; default: false
--gui-testing.setting-output <FILE> 	Save gui settings in the given settings output file
"""

olist = text.split("\n")
slist = []
for x in olist:
    for s in x.split(" "):
        slist.append(s)

met = ""

for x in slist:

    if x.startswith("--") or x.startswith("-"):
        y = 64-len(x)
        z = y//4+(y%4>0)
        if not x[1].isnumeric():
            if len(met) > 1:
                if met[len(met)-1] == '~':
                    met += "#TODO add type\n    " + x + ":" +'\t' * z + "~"
                else:
                    met += "    " + x + ":" + '\t' * z + "~"
            else:
                met += "    " + x + ":" +'\t' * z + "~"
    if x.startswith("<STRING>") or x.startswith("<FILE>") or x.startswith("<PATH>") or x.startswith("<ID>") or x.startswith("<PROJ_DEFINITION>"):
        met += "Optional[str] = None\n"
    if x.startswith("<BOOL>"):
        met += "Optional[bool] = None\n"
    if x.startswith("<FLOAT>") or x.startswith("<TIME>"):
        met += "Optional[float] = None\n"
    if x.startswith("<INT>") or x.startswith("<UINT>"):
        met += "Optional[int] = None\n"
    if x.startswith("<COLOR>"):
        met += "Optional[float] = None  #!!!\n"
    if x.startswith("<2D-POSITION>") or x.startswith("<3D-POSITION>") or x.startswith("<2D-BOUNDING_BOX>"):
        met += "Optional[List[float]] = None  #!!!\n"
    if x.startswith("<POSITION-VECTOR>"):
        met += "Optional[List[List[float]]] = None  #!!!\n"


met = met.replace("--", "").replace(".", "__").replace("-", "_").replace("~", "")
    
with open('flaglist.txt', 'w') as f:
    f.write(met)

