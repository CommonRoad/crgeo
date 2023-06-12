from typing import Dict

from commonroad_geometric.learning.reinforcement.commonroad_gym_env import CommonRoadGymEnv, CommonRoadGymStepInfo


def on_episode_end(env: CommonRoadGymEnv, info: CommonRoadGymStepInfo) -> Dict[str, float]:

    termination_reasons = env.termination_reasons if isinstance(env, CommonRoadGymEnv) else env.get_attr('termination_reasons')[0]
    termination_criteria_flags = dict.fromkeys(termination_reasons, False)
    termination_reason = info.get('termination_reason')
    termination_criteria_flags[termination_reason] = True
    out_dict: Dict[str, float] = {}

    env_reward_component_info = info['reward_component_episode_info']
    assert env_reward_component_info is not None
    for reward_component, component_info in env_reward_component_info.items():
        for component_metric, component_value in component_info.items():
            out_dict[f"reward_{reward_component}_ep_{component_metric}"] = component_value

    vehicle_aggregate_stats = info['vehicle_aggregate_stats']
    assert vehicle_aggregate_stats is not None
    for state, state_info in vehicle_aggregate_stats.items():
        for state_metric, state_value in state_info.items():
            out_dict[f"vehicle_{state}_ep_{state_metric}"] = state_value

    num_obstacles = info.get('total_num_obstacles')
    out_dict["ep_num_obstacles"] = num_obstacles

    cumulative_reward = info.get('cumulative_reward')
    out_dict["ep_cumulative_reward"] = cumulative_reward

    next_reset_ready = info.get('next_reset_ready')
    out_dict["next_reset_ready"] = float(next_reset_ready)

    episode_length = info.get('time_step')
    out_dict["ep_length"] =  episode_length

    for termination_criteria in termination_reasons:
        out_dict["termination_{termination_criteria}"] =  termination_criteria_flags[termination_criteria]

    return out_dict
