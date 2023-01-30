# visualization parameters
basic_shape_parameters_obs = {
    'opacity': 1.0,
    'facecolor': '#1d7eea',
    'edgecolor': '#0066cc',
    'zorder': 20
}

basic_shape_parameters_ego = {
    'opacity': 1.0,
    'facecolor': '#d95558',
    'edgecolor': '#831d20',
    'linewidth': 0.5,
    'zorder': 20
}

draw_params_obstacle = {
    'dynamic_obstacle': {
        'draw_shape': True,
        'draw_icon': False,
        'draw_bounding_box': True,
        'show_label': True,
        'trajectory_steps': 25,
        'zorder': 20,
        'shape': basic_shape_parameters_obs,
        'occupancy': {
            'draw_occupancies':
            1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
            'shape': {
                'opacity': 0.25,
                'facecolor': '#6f9bcb',
                'edgecolor': '#48617b',
                'zorder': 18,
            }
        }
    }
}

draw_params_ego = {
    'dynamic_obstacle': {
        'draw_shape': True,
        'draw_icon': False,
        'draw_bounding_box': True,
        'show_label': False,
        'trajectory_steps': 0,
        'zorder': 20,
        'shape': basic_shape_parameters_ego,
        'occupancy': {
            'draw_occupancies':
            1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
            'shape': {
                'opacity': 0.25,
                'facecolor': '#b05559',
                'edgecolor': '#9e4d4e',
                'zorder': 18,
            }
        }
    }
}
