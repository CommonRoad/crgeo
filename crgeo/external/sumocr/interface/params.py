ego_id_start = 'ego'

id_convention = {
    'lane_ext': 1,
    'lane_special': 2,
    'obstacle_vehicle': 3,
    'ego_vehicle': 4
}

# Minimum angle that is seen as 180°. Used in convert_lanelet_net to detect points on straight line at the end of
# internal junctions. Due to very short road lanes (e.g. length = 0.1) and coordinate accuracy of only two digits
# the deviation of the angle is much higher as machine precision.
straight_angle_dwell = 175

# visualization parameter
basic_shape_parameters_static = {
    'opacity': 1.0,
    'facecolor': '#1d7eea',
    'edgecolor': '#0066cc',
    'zorder': 20
}

basic_shape_parameters_dynamic = {
    'opacity': 1.0,
    'facecolor': '#1d7eea',
    'edgecolor': '#0066cc',
    'zorder': 20
}

draw_params_scenario = {
    'dynamic_obstacle': {
        'draw_shape': True,
        'draw_icon': False,
        'draw_bounding_box': True,
        'show_label': True,
        'trajectory_steps': 25,
        'zorder': 20,
        'occupancy': {
            'draw_occupancies':
            1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
            'shape': {
                'polygon': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'zorder': 18,
                },
                'rectangle': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'zorder': 18,
                },
                'circle': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'zorder': 18,
                }
            },
        },
        'shape': {
            'polygon': basic_shape_parameters_dynamic,
            'rectangle': basic_shape_parameters_dynamic,
            'circle': basic_shape_parameters_dynamic
        },
        'trajectory': {
            'facecolor': '#000000'
        }
    },
    'static_obstacle': {
        'shape': {
            'polygon': basic_shape_parameters_static,
            'rectangle': basic_shape_parameters_static,
            'circle': basic_shape_parameters_static,
        }
    },
    'lanelet_network': {
        'lanelet': {
            'left_bound_color': '#555555',
            'right_bound_color': '#555555',
            'center_bound_color': '#dddddd',
            'draw_left_bound': True,
            'draw_right_bound': True,
            'draw_center_bound': True,
            'draw_border_vertices': False,
            'draw_start_and_direction': True,
            'show_label': True,
            'draw_linewidth': 0.5,
            'fill_lanelet': True,
            'facecolor': '#c7c7c7'
        }
    },
}
