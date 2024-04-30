import os

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_geometric.external.commonroad_route_planner.route_planner import RoutePlanner
from commonroad_geometric.external.commonroad_route_planner.utility.visualization import visualize_route

if __name__ == "__main__":
    # ========== initialization =========== #
    path_scenarios = os.path.join(os.getcwd(), "../scenarios/")
    id_scenario = 'USA_US101-22_1_T-1'
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(f"{path_scenarios}{id_scenario}.xml").open()
    # retrieve the first planning problem in the problem set
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # ========== route planning =========== #
    # instantiate a route planner with the scenario and the planning problem
    route_planner = RoutePlanner(
        scenario,
        planning_problem,
        backend=RoutePlanner.Backend.NETWORKX_REVERSED,
        reach_goal_state=False)
    # plan routes, and save the routes in a route candidate holder
    candidate_holder = route_planner.plan_routes()

    # ========== retrieving routes =========== #
    # option 1: retrieve all routes
    list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
    print(f"Number of route candidates: {num_route_candidates}")
    # here we retrieve the first route in the list, this is equivalent to: route = list_routes[0]
    route = candidate_holder.retrieve_first_route()

    # option 2: retrieve the best route by orientation metric
    # route = candidate_holder.retrieve_best_route_by_orientation()

    # ========== visualization =========== #
    visualize_route(route, draw_route_lanelets=True, draw_reference_path=True)
