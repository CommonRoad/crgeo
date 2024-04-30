from typing import Any, Dict, Optional, Tuple, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_scenario(
    scenario: Union[Scenario, LaneletNetwork],
    planning_problem_set: Optional[PlanningProblemSet] = None,
    show: bool = False,
    title: bool = True,
    ax: Optional[Axes] = None,
    draw_params: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> Tuple[Figure, Axes]:
    from matplotlib import pyplot as plt
    from commonroad.visualization.mp_renderer import MPRenderer
    rnd = MPRenderer(ax=ax, figsize=figsize)

    if draw_params is None:
        draw_params = {
            'time_begin': 0,
            'lanelet': {
                'show_label': True,
                'facecolor': 'black'
            },
            'intersection': {
                'draw_intersections': False,
                'show_label': False
            }
        }

    scenario.draw(rnd, draw_params=draw_params)
    # plot the planning problem set
    if planning_problem_set is not None:
        planning_problem_set.draw(rnd)
    rnd.render()
    if title and isinstance(scenario, Scenario):
        if scenario.location is not None:
            coordinates = f"{scenario.location.gps_latitude:.6f}, {scenario.location.gps_longitude:.6f}"
            plt.title(str(scenario.scenario_id) + ' (' + coordinates + ')')
        else:
            plt.title(str(scenario.scenario_id))

    if show:
        plt.show()

    return rnd.f, rnd.ax
