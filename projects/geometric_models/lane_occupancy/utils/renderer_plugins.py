from typing import List, Optional, Set
import torch
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
#import seaborn

from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.math import calc_closest_factors
from commonroad_geometric.common.utils.system import debugger_is_active
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.color_utils import cmyk_to_rgb, rgb_to_cmyk
from commonroad_geometric.rendering.defaults import ColorTheme
from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import draw_lanelet
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D

np.set_printoptions(formatter={'float': lambda x: "{0:0.4e}".format(x)})
matplotlib.rcParams['hatch.linewidth'] = 0.7

plt.style.use('ggplot')
#plt.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
#plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True) #RAISES FILENOTFOUNDERROR
plt.rc('xtick', labelsize=39)
plt.rc('ytick', labelsize=39)
plt.rc('axes', labelsize=47)
plt.rc('legend',fontsize=23)

# TODO Needs cleanup: Lots of duplicate code, inconsistent responsibilities, matplotlib hacks

class RenderLaneletOccupancyPredictionPlugin(BaseRendererPlugin):
    _suppress_warnings = False

    def __init__(
        self,
        included_lanelet_ids: Optional[Set[int]] = None,
        enable_matplotlib_plot: bool = True,
        plot_freq: int = 10
    ) -> None:
        self.enable_matplotlib_plot = enable_matplotlib_plot
        self.plot_freq = plot_freq
        self.included_lanelet_ids = included_lanelet_ids
        self.call_count = 0
        self.plot_count = 0

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if not hasattr(self, 'call_count'):
            self.call_count = 0
        self.call_count += 1

        try:
            output_pred = params.data.l.occupancy_predictions
            output_enc = params.data.l.occupancy_encodings
            if isinstance(output_pred, tuple):
                joint_occ_probs = output_pred[0]
            else:
                joint_occ_probs = output_pred
            if isinstance(output_enc, tuple):
                z = output_enc[0]
            else:
                z = output_enc
        except Exception:
            output_pred = params.render_kwargs.get('output')
            if output_pred is None:
                if not RenderLaneletOccupancyPredictionPlugin._suppress_warnings:
                    warnings.warn("RenderLaneletOccupancyPredictionPlugin cannot render because it could not retrieve predictions. Future warnings will be suppressed.")
                RenderLaneletOccupancyPredictionPlugin._suppress_warnings = True
                return
            z = output_pred[0]
            joint_occ_probs, info = output_pred[-1]

        # NUM_LANELETS x TIME_HORIZON x RESOLUTION
        joint_occ_probs = joint_occ_probs.squeeze(0)

        num_lanelets = joint_occ_probs.shape[0]
        time_horizon = joint_occ_probs.shape[1]
        resolution = joint_occ_probs.shape[2]

        lane_timespace = torch.linspace(
            0, 
            1, 
            time_horizon,
            dtype=torch.float32
        )[None, :, None].repeat(num_lanelets, 1, resolution)
        lane_linspace = torch.linspace(
            0, 
            1, 
            resolution,
            dtype=torch.float32
        )[None, None, :].repeat(num_lanelets, time_horizon, 1)

        road_color_blue = (joint_occ_probs*lane_timespace).max(axis=1)[0]
        road_color_red = (joint_occ_probs*(1-lane_timespace)).max(axis=1)[0]

        for lanelet_idx, lanelet_id_th in enumerate(params.data.lanelet.id):
            lanelet_id = lanelet_id_th.item()
            if self.included_lanelet_ids is not None and lanelet_id not in self.included_lanelet_ids:
                continue

            center_vertices = params.simulation.lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices
            center_vertices = center_vertices[~(center_vertices == 0.0).all(axis=1)]
            
            lanelet_centerline = params.simulation.get_lanelet_center_polyline(lanelet_id, vertices=center_vertices)

            for t in range(resolution):
                if road_color_red[lanelet_idx, t].item() > 0.05 or road_color_blue[lanelet_idx, t].item() > 0.05:
                    color = (
                        road_color_red[lanelet_idx, t].item(),
                        0.0,
                        road_color_blue[lanelet_idx, t].item(), 
                        1.0
                    )
            
                    path_var = lanelet_centerline.length*t/(resolution - 1)
                    pos = lanelet_centerline(path_var)

                    viewer.draw_circle(
                        origin=pos,
                        radius=0.8,
                        color=color,
                        outline=False,
                        linecolor=(0.1,1.0,0.1),
                        linewidth=None
                    )

        if not hasattr(self, 'enable_matplotlib_plot'):
            self.enable_matplotlib_plot = True
        if not hasattr(self, 'plot_freq'):
            self.plot_freq = 10
        if not hasattr(self, 'plot_count'):
            self.plot_count = 0

        if not self.enable_matplotlib_plot:
            return

        # TODO: less hardcoding, cleanup
        
        selected_lanelet_idx = self.plot_count % num_lanelets
        selected_lanelet_id = params.data.lanelet.id[selected_lanelet_idx].item()
        frame_interval = 5
        selected_lanelet = params.simulation.find_lanelet_by_id(selected_lanelet_id)
            
        # highlight focused lanelet
        color = (1.0, 1.0, 1.0, 0.5) if viewer.theme == ColorTheme.DARK else (0.0, 0.8, 0.0, 1.0)

        draw_lanelet(
            viewer=viewer,
            left_vertices=selected_lanelet.left_vertices,
            center_vertices=selected_lanelet.center_vertices,
            right_vertices=selected_lanelet.right_vertices,
            color=color,
            linewidth=1.0,
            font_size=14,
            label=None
        )

        if self.call_count % self.plot_freq != 0:
            return
        import matplotlib.pyplot as plt
        import matplotlib
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4e}".format(x)})
        matplotlib.rcParams['hatch.linewidth'] = 0.7
        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        #plt.rc('font', family='serif', serif='Times')
        #plt.rc('text', usetex=True) #RAISES FILENOTFOUNDERROR
        plt.rc('xtick', labelsize=21)
        plt.rc('legend',fontsize=21)
        plt.rc('ytick', labelsize=21)
        plt.rc('axes', labelsize=26)

        # try:
        #     user_input = input("Please enter lane index to plot: (Enter q to disable)")
        #     if user_input.lower() == 'q':
        #         plots = False
        #         continue_plotting = False
        #         print("Disabled plotting")
        #     else:
        #         lanelet_idx = int(user_input)
        #         continue_plotting = True
        # except:
        #     continue_plotting = False
        #if continue_plotting:
        try:
            occ_prob_components = output_pred[1][1]['occ_prob_components'].squeeze(0)
        except KeyError:
            occ_prob_components = output_pred[1]['occ_prob_components'].squeeze(0)
        n_distr = occ_prob_components.shape[-1]

        plt.cla()
        fig, ax = plt.subplots(figsize=(8, 8))
        #frames = list(range(time_horizon))
        frames = list(range(20))
        def draw_frame(frame: int): #, joint_occ_probs, occ_probs):
            legend = []
            #for frame in frames:
            ax.clear()
            ax.set_ylim(0, 1)
            ax.set_xlim(0, params.data.lanelet.lanelet_length[selected_lanelet_idx].item())
            #ax.set_title(f"t+{frame+1}")
            #ax.set_title(f"Joint lanelet occupancy predictions")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("p(x)")

            for t in range(0, time_horizon, frame_interval):

                x = (params.data.lanelet.lanelet_length[selected_lanelet_idx]*lane_linspace[selected_lanelet_idx, t, :]).detach().numpy()
                y = joint_occ_probs[selected_lanelet_idx, t, :].detach().numpy()
                color = (1 - t/time_horizon, 0, t/time_horizon)
                ax.plot(x, y, color=color, linewidth=1.5)

                # print(t, color, frame, x.shape, y.shape, y.mean())
                
                time = params.scenario.dt * t
                
                # for c in range(n_distr):
                #     y_c = occ_prob_components[selected_lanelet_idx, t, :, c].detach().numpy()
                #     ax.plot(x, y_c, linestyle='dotted', color=color, linewidth=1.0)
                
                legend.append(f'{time:.1f}s')

            ax.legend(legend)
            return [ax]
                #
                #plt.show()
                #ax.vlines(mu_t[frame, 0, :], ymin=1-0.1*w_arr, ymax=1+0.1*w_arr)
                #ax.vlines(data.pos.ravel(), ymin=0.0, ymax=0.1, colors='green')

            #animation = FuncAnimation(fig, draw_frame, 20, blit=True)
            #writergif = PillowWriter(fps=len(frames)/7.5)
        ax = draw_frame(0)
        figure_dir = os.path.join('tutorials', 'output', 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        plt.savefig(os.path.join(figure_dir, f'occpred_{self.call_count}.pdf'))
        plt.close()
        plt.clf()
        self.plot_count += 1

        grid_height, grid_width = calc_closest_factors(z.shape[1])
        selected_z = z[selected_lanelet_idx].view(grid_height, grid_width).detach().cpu().numpy()
        plt.grid(None)
        plt.axis('off')
        cmap = LinearSegmentedColormap.from_list('BlackGreen', [(0, 0, 0), (0, 1, 0)], N=2000)
        heatmap = plt.imshow(selected_z, cmap=cmap)
        cbar = plt.colorbar(heatmap, orientation="horizontal", pad=0.1)
        plt.rcParams["axes.grid"] = False
        plt.savefig(os.path.join(figure_dir, f'occenc_{self.call_count}.pdf'))
        plt.grid(None)
        plt.axis('off')
        plt.close()
        plt.clf()
        plt.cla()



        # animation = FuncAnimation(fig, draw_frame, 20, blit=True)
        # writergif = PillowWriter(fps=len(frames)/7.5)
        # animation.save(f'movingdistr.gif', writer=writergif)




class RenderLaneletEncodingPlugin(BaseRendererPlugin):
    _suppress_warnings = False

    def __init__(
        self,
        grid_size: int = 20,
        alpha: float = 0.5,
        compute_max: bool = False,
        multiplier: float = 3.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        included_lanelet_ids: Optional[Set[int]] = None,
        ego_lanelet_color: bool = True

    ) -> None:
        self.grid_size = grid_size
        self.alpha = alpha
        self.compute_max = compute_max
        self.multiplier = multiplier
        self.offset = np.array([x_offset, y_offset])
        self.included_lanelet_ids = included_lanelet_ids
        self.ego_lanelet_color = ego_lanelet_color

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:

        has_ego_lanelet = params.ego_vehicle is not None
        if has_ego_lanelet:
            ego_lanelet_assignments = [lid for lid in params.simulation.obstacle_id_to_lanelet_id[params.ego_vehicle.obstacle_id] if lid in params.ego_vehicle.ego_route.lanelet_id_route]
            if len(ego_lanelet_assignments) > 0:
                ego_lanelet_id = ego_lanelet_assignments[0]
            else:
                ego_lanelet_id = None
                has_ego_lanelet = False
        else:
            ego_lanelet_id = None


        try:
            output = params.data.l.occupancy_encodings
            if isinstance(output, tuple):
                z = output[0]
            else:
                z = output
        except Exception:
            output = params.render_kwargs.get('output')
            if output is None:
                if not RenderLaneletEncodingPlugin._suppress_warnings:
                    warnings.warn("RenderLaneletEncodingPlugin cannot render because it could not retrieve encodings. Future warnings will be suppressed.")
                RenderLaneletEncodingPlugin._suppress_warnings = True
                return
            z, (joint_occ_probs, info) = output

        self.dim_reduction = z.shape[1] // self.grid_size
        binned_z = z[:, :(z.shape[1] // self.dim_reduction)*self.dim_reduction].reshape(z.shape[0], z.shape[1] // self.dim_reduction, -1)
        if self.compute_max:
            z_agg = torch.max(binned_z, 2)[0]
        else:
            z_agg = torch.mean(binned_z, 2)

        for lanelet_idx, lanelet_id_th in enumerate(params.data.lanelet.id):
            lanelet_id = lanelet_id_th.item()
            if self.included_lanelet_ids is not None and lanelet_id not in self.included_lanelet_ids:
                continue
            lanelet_centerline = params.simulation.get_lanelet_center_polyline(lanelet_id)

            is_ego_lanelet = has_ego_lanelet and lanelet_id == ego_lanelet_id

            length = lanelet_centerline.length/z_agg.shape[1]
            width = 2.2
            vertices = np.array([
                [- 0.5 * length, - 0.5 * width], [- 0.5 * length, + 0.5 * width],
                [+ 0.5 * length, + 0.5 * width], [+ 0.5 * length, - 0.5 * width],
                [- 0.5 * length, - 0.5 * width]
            ])

            for t in range(z_agg.shape[1]):

                z_value = 0.5 + self.multiplier*(z_agg[lanelet_idx, t].item())
                z_value = np.clip(z_value, 0, 1)

                if z_value < 0.25: # don't draw black circles
                    continue

                if is_ego_lanelet and self.ego_lanelet_color:
                    color = color = (
                        z_value,
                        z_value,
                        z_value, 
                        0.8
                    )
                else:
                    color = (
                        0.0,
                        z_value,
                        0.0,
                        self.alpha
                    )
            
                arclength = lanelet_centerline.length*t/(z_agg.shape[1] - 1)
                pos = lanelet_centerline(arclength) + self.offset
                orientation = lanelet_centerline.get_direction(arclength)

                viewer.draw_shape(
                    vertices=vertices,
                    position=pos,
                    angle=orientation,
                    filled=True,
                    color=color
                )





class RenderPathOccupancyPredictionPlugin(BaseRendererPlugin):
    def __init__(
        self,
        render_ego_encoding: bool = True,
        render_lanelet_encodings: bool = False,
        render_message_intensitites: bool = False,
        encoding_grid_size: Optional[int] = None,
        encoding_alpha: float = 1.0,
        encoding_compute_max: bool = False,
        encoding_multiplier: float = 0.6,
        encoding_offset: float = 3.5,
        encoding_offset_ego: float = 5.0,
        multi_color: bool = True,
        skip_frames: bool = False,
        draw_lanelet_indicators: bool = False,
        enable_plots: bool = False,
        show_plots_debugger: bool = False,
        plot_subcomponents: bool = False,
        plot_freq: int = 10,
        render_ego_vehicle: bool = False,
        fill_path: bool = False,
        border_path: bool = True,
        lanelet_highlighting: bool = False,
        occupancy_circles: bool = False,
        plot_prob_threshold_mean: float = 0.03,
        plot_prob_threshold_max: float = 0.5,
        occ_render_threshold: float = 0.08

    ) -> None:
        self.render_ego_encoding = render_ego_encoding
        self.render_lanelet_encodings = render_lanelet_encodings
        self.render_message_intensities = render_message_intensitites
        self.grid_size = encoding_grid_size
        self.alpha = encoding_alpha
        self.skip_frames = skip_frames
        self.compute_max = encoding_compute_max
        self.multiplier = encoding_multiplier
        self.offset = encoding_offset
        self.offset_ego = encoding_offset_ego
        self.multi_color = multi_color
        self.draw_lanelet_indicators = draw_lanelet_indicators
        self.enable_plots = enable_plots #or debugger_is_active()
        self.show_plots_debugger = show_plots_debugger #or debugger_is_active()
        self.plot_subcomponents = plot_subcomponents
        self.plot_freq = plot_freq
        self.render_ego_vehicle = render_ego_vehicle
        self.fill_path = fill_path
        self.border_path = border_path
        self.lanelet_highlighting = lanelet_highlighting
        self.occupancy_circles = occupancy_circles
        self.plot_prob_threshold_mean = plot_prob_threshold_mean
        self.plot_prob_threshold_max = plot_prob_threshold_max
        self.occ_render_threshold = occ_render_threshold
        self.call_count = 0
        self.plot_count = 0
        self.last_frame_index = 0

        self.figure_dir = os.path.join('tutorials', 'output', 'figures')
        # if debugger_is_active():
        #     shutil.rmtree(self.figure_dir, ignore_errors=True)

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if not hasattr(self, 'call_count'):
            self.call_count = 0
        self.call_count += 1
        should_create_plot = (self.enable_plots or self.show_plots_debugger) and viewer.frame_count % self.plot_freq == 0

        try:
            output_pred = params.data.ego_route_occupancy_predictions
            output_enc = params.data.z_ego_route
            message_intensities = params.data.message_intensities
            if isinstance(output_pred, tuple):
                joint_occ_probs = output_pred[0]
            else:
                joint_occ_probs = output_pred
            if isinstance(output_enc, tuple):
                z = output_enc[0]
            else:
                z = output_enc
        except Exception:
            output_pred = params.render_kwargs.get('output')
            if output_pred is None:
                if not RenderLaneletOccupancyPredictionPlugin._suppress_warnings:
                    warnings.warn("RenderLaneletOccupancyPredictionPlugin cannot render because it could not retrieve predictions. Future warnings will be suppressed.")
                RenderLaneletOccupancyPredictionPlugin._suppress_warnings = True
                return
            z = output_pred[0]
            joint_occ_probs, info = output_pred[-1]

        if should_create_plot:
            should_create_plot = should_create_plot and joint_occ_probs.max().item() > self.plot_prob_threshold_max and joint_occ_probs.mean().item() > self.plot_prob_threshold_mean
            if not should_create_plot and self.skip_frames: 
                viewer.skip_frames(n=1)
                print(f"Skip frames... too low density (max={joint_occ_probs.max().item():.4f}, mean={joint_occ_probs.mean().item():.4f}")
                return

        # y_cont = params.data.lanelet.occupancy_continuous.view(
        #     params.data.lanelet.occupancy_continuous.shape[0], params.data.lanelet.occupancy_time_horizon, params.data.lanelet.occupancy_max_vehicle_count, 3
        # )
        if isinstance(z, tuple):
            z = z[0]
        if z.ndim == 1:
            z = z.unsqueeze(0)

        if self.render_message_intensities:
            message_intensitiy_means = message_intensities.mean(dim=1)
            norm_message_intensity_means = (message_intensitiy_means - message_intensitiy_means.mean()) / message_intensitiy_means.std()
            sigmoid_intensity_means = torch.sigmoid(norm_message_intensity_means)
            for edge_idx in range(norm_message_intensity_means.shape[0]):
                source_lanelet_idx = params.data.l2l.edge_index[0, edge_idx].item()
                target_lanelet_idx = params.data.l2l.edge_index[1, edge_idx].item()
                source_lanelet_id = params.data.lanelet.id[source_lanelet_idx].item()
                target_lanelet_id = params.data.lanelet.id[target_lanelet_idx].item()
                source_lanelet_centerline = params.simulation.get_lanelet_center_polyline(source_lanelet_id)
                target_lanelet_centerline = params.simulation.get_lanelet_center_polyline(target_lanelet_id)
                source_pos = source_lanelet_centerline(source_lanelet_centerline.length / 2)
                target_pos = target_lanelet_centerline(target_lanelet_centerline.length / 2)

                viewer.draw_line(
                    source_pos,
                    target_pos,
                    linewidth=4,
                    color=(
                        sigmoid_intensity_means[edge_idx].item(),
                        sigmoid_intensity_means[edge_idx].item(),
                        sigmoid_intensity_means[edge_idx].item(),
                        1.0
                    ),
                )
                
        
        # 1 x TIME_HORIZON x RESOLUTION
        time_horizon = joint_occ_probs.shape[1]
        resolution = joint_occ_probs.shape[2]
        
        linspace = torch.linspace(
            0, 
            1, 
            time_horizon,
            dtype=torch.float32,
            device=joint_occ_probs.device
        ).unsqueeze(-1).repeat(1, resolution).unsqueeze(0)

        road_color_blue = (joint_occ_probs*linspace).max(axis=1)[0]
        road_color_red = (joint_occ_probs*(1-linspace)).max(axis=1)[0]

        path_length = params.data.path_length
        arclengths = np.arange(resolution) * path_length / resolution # TODO linspace instead...
        if not hasattr(params.data, 'cumulative_prior_length_abs'):
            # hack to avoid OccupancyEncodingPostProcessor crash due to not storing tensors # TODO fix, remove deepcopy dep.
            from projects.geometric_models.lane_occupancy.utils import preprocess_conditioning
            preprocess_conditioning(
                data=params.data,
                walks=params.data.walks,
                walk_start_length=params.data.walk_start_length,
                path_length=path_length,
                walk_masks=params.data.ego_trajectory_sequence_mask.bool()
            )
        n_lanelets = params.data.cumulative_prior_length_abs.shape[0]

        lanelet_counter = -1
        
        try:
            integration_domains_masked = params.data.integration_domains_positive_path
            integration_domains_masked_neg = params.data.integration_domains_negative_path
        except AttributeError:
            integration_domains_masked, integration_domains_masked_neg = None, None
        
        # print('n occ slots', integration_domains_masked.shape[0], '/', integration_domains_masked_neg.shape[0], 'loss', params.data.positive_integrals_path.mean().item(), params.data.negative_integrals_path.mean().item())
        
        merged_centerline_input: List[ContinuousPolyline] = []
        merged_leftline_input: List[ContinuousPolyline] = []
        merged_rightline_input: List[ContinuousPolyline] = []

        for t, arclength in enumerate(arclengths):
            if lanelet_counter < n_lanelets - 1:
                if lanelet_counter < 0 or arclength > params.data.cumulative_prior_length_abs[lanelet_counter + 1]:
                    lanelet_counter += 1
                    lanelet_idx = params.data.walks[lanelet_counter].item()
                    lanelet_id = params.data.lanelet.id[lanelet_idx]
                    lanelet_centerline = params.simulation.get_lanelet_center_polyline(lanelet_id.item())
                    lanelet_leftline = params.simulation.get_lanelet_left_polyline(lanelet_id.item())
                    lanelet_rightline = params.simulation.get_lanelet_right_polyline(lanelet_id.item())

                    # lanelet_length = params.data.l.length[lanelet_idx].item()
                    # prior_length_rel = params.data.cumulative_prior_length[lanelet_counter].item()
                    # integration_lower_limits_rel = params.data.integration_lower_limits[lanelet_counter].item()
                    # integration_upper_limits_rel = params.data.integration_upper_limits[lanelet_counter].item()
                    # prior_length_abs = params.data.cumulative_prior_length_abs[lanelet_counter].item()
                    integration_lower_limits_abs = params.data.integration_lower_limits_abs[lanelet_counter].item()
                    integration_upper_limits_abs = params.data.integration_upper_limits_abs[lanelet_counter].item()
                    merged_centerline_input.append(lanelet_centerline.slice(integration_lower_limits_abs, integration_upper_limits_abs, inclusive=False))
                    merged_leftline_input.append(lanelet_leftline.slice(integration_lower_limits_abs, integration_upper_limits_abs, inclusive=False))
                    merged_rightline_input.append(lanelet_rightline.slice(integration_lower_limits_abs, integration_upper_limits_abs, inclusive=False))

        try:
            merged_centerline = ContinuousPolyline.merge(*merged_centerline_input)
            merged_leftline = ContinuousPolyline.merge(*merged_leftline_input)
            merged_rightline = ContinuousPolyline.merge(*merged_rightline_input)
        except ValueError as e:
            print(e)
            return

        
        # fill path
        # viewer.draw_polygon(
        #     v=np.vstack([merged_leftline.waypoints, merged_rightline.waypoints[::-1], merged_leftline.waypoints[:1]]),
        #     color=(0.1, 0.1, 0.1, 0.3),
        # )
        width = merged_leftline.get_lateral_distance(merged_rightline(0)) - 0.15
        merged_leftline = merged_centerline.lateral_translate(distance=-width/2)
        merged_rightline = merged_centerline.lateral_translate(distance=width/2)

        if self.fill_path:
            # color = (0.0, 0.0, 0.0, 1.0) if viewer.theme == ColorTheme.DARK else (0.825, 0.8, 0.825, 0.4)
            # color = (0.0, 0.0, 0.0, 1.0) if viewer.theme == ColorTheme.DARK else (0.8, 0.8, 0.8, 0.6)
            color = (0.0, 0.0, 0.0, 1.0) if viewer.theme == ColorTheme.DARK else (0.8, 0.95, 0.8, 1.0)
            path_fill_res = 100
            length = merged_centerline.length/path_fill_res

            for t in range(path_fill_res):
                arclength = merged_centerline.length*t/(path_fill_res - 1)
                pos = merged_centerline(arclength)
                vertices = np.array([
                    [- 0.5 * length, - 0.5 * width], [- 0.5 * length, + 0.5 * width],
                    [+ 0.5 * length, + 0.5 * width], [+ 0.5 * length, - 0.5 * width],
                    [- 0.5 * length, - 0.5 * width]
                ])
                orientation = merged_centerline.get_direction(arclength)
                viewer.draw_shape(
                    vertices=vertices,
                    position=pos,
                    angle=orientation,
                    filled=True,
                    color=color,
                    linewidth=0,
                    border=False,
                    index=0
                )

        if self.border_path:
            if self.fill_path:
                #color = (0.6, 0.6, 0.6, 1.0) if viewer.theme == ColorTheme.DARK else (0.75, 0.75, 0.75, 1.0)
                color = (0.6, 0.6, 0.6, 1.0) if viewer.theme == ColorTheme.DARK else (0.0, 0.0, 0.0, 1.0)
            else:
                color = (0.6, 0.6, 0.6, 1.0) if viewer.theme == ColorTheme.DARK else (0.0, 0.8, 0.0, 1.0)
            linewidth = 0.7 if viewer.theme == ColorTheme.DARK else 1.0
            viewer.draw_polyline(
                v=merged_leftline.waypoints,
                linewidth=linewidth,
                color=color,
            )
            viewer.draw_polyline(
                v=merged_rightline.waypoints,
                linewidth=linewidth,
                color=color,
            )

            viewer.draw_line(
                merged_rightline.waypoints[0],
                merged_leftline.waypoints[0],
                linewidth=linewidth,
                color=color,
            )
            viewer.draw_line(
                merged_rightline.waypoints[-1],
                merged_leftline.waypoints[-1],
                linewidth=linewidth,
                color=color,
            )

        if self.render_ego_vehicle:
            viewer.draw_shape(
                np.array([
                    [-2.4, -1.],
                    [-2.4, 1.],
                    [2.4, 1.],
                    [2.4, -1.],
                    [-2.4, -1.]
                ]),
                merged_centerline(0),
                merged_centerline.get_direction(0),
                filled=True,
                linewidth=0.4,
                fill_color=(1.0, 1.0, 1.0, 0.5),
                border=(1.0, 1.0, 1.0, 1.0)
            )

        lanelet_counter = -1
        for t, arclength in enumerate(arclengths):
            if lanelet_counter < n_lanelets - 1:
                if lanelet_counter < 0 or arclength > params.data.cumulative_prior_length_abs[lanelet_counter + 1]:
                    lanelet_counter += 1
                    lanelet_idx = params.data.walks[lanelet_counter].item()
                    lanelet_id = params.data.lanelet.id[lanelet_idx]
                    lanelet_centerline = params.simulation.get_lanelet_center_polyline(lanelet_id.item())
                    lanelet_leftline = params.simulation.get_lanelet_left_polyline(lanelet_id.item())
                    lanelet_rightline = params.simulation.get_lanelet_right_polyline(lanelet_id.item())

                    #print('draw_lanelet', lanelet_id, lanelet_length, prior_length_rel, prior_length_abs, integration_lower_limits_abs, 
                    # integration_upper_limits_abs, integration_lower_limits_rel, integration_upper_limits_rel
                        
                    # highlight focused lanelet
                    if self.lanelet_highlighting:
                        draw_lanelet(
                            viewer=viewer,
                            left_vertices=lanelet_leftline.slice(integration_lower_limits_abs, integration_upper_limits_abs),
                            center_vertices=lanelet_centerline.slice(integration_lower_limits_abs, integration_upper_limits_abs),
                            right_vertices=lanelet_rightline.slice(integration_lower_limits_abs, integration_upper_limits_abs),
                            color=(1.0, 1.0, 1.0, 0.2),
                            linewidth=0.5,
                            font_size=14,
                            label=None
                        )
                        viewer.draw_circle(
                            origin=lanelet_leftline(integration_lower_limits_abs),
                            radius=0.25,
                            color=(1.0, 0.0, 0.0, 1.0),
                            outline=False,
                            linecolor=(1.0,0.0,0.0),
                            linewidth=None
                        )
                        viewer.draw_circle(
                            origin=lanelet_rightline(integration_upper_limits_abs),
                            radius=0.25,
                            color=(0.0, 0.0, 1.0, 1.0),
                            outline=False,
                            linecolor=(0.0,0.0,0.0),
                            linewidth=None
                        )

                    #y_cont_lanelet = y_cont[lanelet_idx, 0, :, :]
                    #y_cont_lanelet_active = y_cont_lanelet[y_cont_lanelet[:, 2] == 1.0, :]

                    # for y_cont_idx in range(y_cont_lanelet_active.shape[0]):
                    #     start_arclength = y_cont_lanelet_active[y_cont_idx, 0].item() 
                    #     end_arclength = y_cont_lanelet_active[y_cont_idx, 1].item()
                    #     start_pos = lanelet_centerline(lanelet_centerline.length * start_arclength)
                    #     end_pos = lanelet_centerline(lanelet_centerline.length * end_arclength)

                    #     start_arclength_global = start_arclength * params.data.lanelet.lanelet_length[lanelet_idx].item() / path_length - params.data.integration_lower_limits[lanelet_counter].item() + params.data.cumulative_prior_length[lanelet_counter].item()
                    #     end_arclength_global = end_arclength * params.data.lanelet.lanelet_length[lanelet_idx].item() / path_length - params.data.integration_lower_limits[lanelet_counter].item() + params.data.cumulative_prior_length[lanelet_counter].item()

                        #print(y_cont_lanelet_active.shape[0], start_arclength_global, end_arclength_global)

                        # viewer.draw_circle(
                        #     origin=start_pos,
                        #     radius=1.5,
                        #     color=(1.0, 0.0, 0.0, 1.0),
                        #     outline=False,
                        #     linecolor=(0.1,1.0,0.1),
                        #     linewidth=None
                        # )
                        # viewer.draw_circle(
                        #     origin=end_pos,
                        #     radius=1.5,
                        #     color=(0.0, 0.0, 1.0, 1.0),
                        #     outline=False,
                        #     linecolor=(0.1,1.0,0.1),
                        #     linewidth=None
                        # )


            if not viewer.pretty and max(road_color_red[0, t].item(), road_color_blue[0, t].item()) < self.occ_render_threshold:
                continue
                
            center_arclength =  merged_centerline.length*t/resolution
            pos = merged_centerline(center_arclength)

            length = path_length/resolution
            top_arclength = center_arclength + length
            bottom_arclength = center_arclength
            width = merged_leftline.get_lateral_distance(merged_rightline(center_arclength)) - 0.1
            top_left = merged_centerline.lateral_translate_point(top_arclength, -width/2) - pos
            top_right = merged_centerline.lateral_translate_point(top_arclength, width/2) - pos
            bottom_left = merged_centerline.lateral_translate_point(bottom_arclength, -width/2) - pos
            bottom_right = merged_centerline.lateral_translate_point(bottom_arclength, width/2) - pos


            vertices = np.array([
                bottom_left, top_left,
                top_right, bottom_right,
                bottom_left
            ])
            #orientation = lanelet_centerline.get_direction(path_var)

            rgb_color = (
                road_color_red[0, t].item(),
                0.0,
                road_color_blue[0, t].item()
            )

            if viewer.theme == ColorTheme.DARK:
                if self.occupancy_circles:
                    viewer.draw_circle(
                        origin=pos,
                        radius=1.2,
                        color=rgb_color,
                        outline=False,
                        linecolor=(0.1, 1.0, 0.1),
                        linewidth=None
                    )
                else:
                    viewer.draw_shape(
                        vertices=vertices,
                        filled=True,
                        color=rgb_color,
                        linewidth=0,
                        border=False,
                        index=0
                    )
            else:
                c, m, y, k = rgb_to_cmyk(*rgb_color)
                k = 0
                rgb_color = cmyk_to_rgb(c, m, y, k)

                if self.occupancy_circles:
                    rgb_color = (rgb_color[0], rgb_color[1], rgb_color[2], 0.5)
                    viewer.draw_circle(
                        origin=pos,
                        radius=1.2,
                        color=rgb_color,
                        outline=False,
                        linecolor=(0.1, 1.0, 0.1),
                        linewidth=0.0,
                        border=False
                    )
                else:
                    viewer.draw_shape(
                        vertices=vertices,
                        filled=True,
                        color=rgb_color,
                        linewidth=0,
                        border=False,
                        index=0
                    )

            if not hasattr(self, 'draw_lanelet_indicators'):
                self.draw_lanelet_indicators = False
            if self.draw_lanelet_indicators:
                if t == 0:
                    viewer.draw_circle(
                        origin=pos,
                        radius=0.25,
                        color=(1.0, 0.0, 0.0, 1.0),
                        outline=False,
                        linecolor=(0.1,1.0,0.1),
                        linewidth=None
                    )
                elif t == len(arclengths) - 1:
                    viewer.draw_circle(
                        origin=pos,
                        radius=0.25,
                        color=(0.0, 0.0, 1.0, 1.0),
                        outline=False,
                        linecolor=(0.1,1.0,0.1),
                        linewidth=None
                    )
        
        if integration_domains_masked is not None:
            # for i, domain in enumerate(integration_domains_masked_neg):
            #     line = merged_centerline._path_coords(params.data.path_length*domain)
            #     viewer.draw_polyline(
            #         v=line,
            #         linewidth=0.2,
            #         color=(1.0, 0.0, 0.0, 1.0),
            #     )
            for i, domain in enumerate(integration_domains_masked):
                line = merged_centerline._path_coords(params.data.path_length*domain)
                viewer.draw_polyline(
                    v=line,
                    linewidth=0.2,
                    color=(0.0, 1.0, 0.0, 1.0),
                )

        if self.render_lanelet_encodings:
            z = params.data.lanelet.occupancy_encodings
            if self.grid_size is not None and self.grid_size > z.shape[1]:
                self.dim_reduction = z.shape[1] // self.grid_size
                binned_z = z[:, :(z.shape[1] // self.dim_reduction)*self.dim_reduction].reshape(z.shape[0], z.shape[1] // self.dim_reduction, -1)
                if self.compute_max:
                    z_agg = torch.max(binned_z, 2)[0]
                else:
                    z_agg = torch.mean(binned_z, 2)
            else:
                z_agg = z

            for lanelet_idx, lanelet_id_th in enumerate(params.data.lanelet.id):
                lanelet_id = lanelet_id_th.item()
                lanelet_centerline = params.simulation.get_lanelet_center_polyline(lanelet_id)
                draw_line = lanelet_centerline.lateral_translate(self.offset)

                length = lanelet_centerline.length/z_agg.shape[1]
                width = 2.2
                vertices = np.array([
                    [- 0.5 * length, - 0.5 * width], [- 0.5 * length, + 0.5 * width],
                    [+ 0.5 * length, + 0.5 * width], [+ 0.5 * length, - 0.5 * width],
                    [- 0.5 * length, - 0.5 * width]
                ])

                for t in range(z_agg.shape[1]):
                    arclength = draw_line.length*t/(z_agg.shape[1] - 1)
                    pos = draw_line(arclength)
                    orientation = draw_line.get_direction(arclength)

                    if viewer.theme == ColorTheme.DARK:
                        if self.multi_color:
                            z_value = self.multiplier*(z_agg[lanelet_idx, t].item())
                            color = (
                                min(1.0, abs(min(z_value, 0.0))),
                                min(1.0, max(z_value, 0.0)),
                                0.0,
                                self.alpha*0.8
                            )
                        else:
                            z_value = 0.5 + self.multiplier*(z_agg[lanelet_idx, t].item())
                            z_value = np.clip(z_value, 0, 1)

                            if z_value < 0.25: # don't draw black circles
                                continue

                            color = (
                                0.0,
                                z_value,
                                0.0,
                                self.alpha*0.8
                            )

                        viewer.draw_shape(
                            vertices=vertices,
                            position=pos,
                            angle=orientation,
                            filled=True,
                            color=color,
                            linewidth=0,
                            border=False
                        )

                    else:
                        viewer.draw_shape(
                            vertices=vertices,
                            position=pos,
                            angle=orientation,
                            filled=True,
                            color=(
                                1.0,
                                0.0,
                                0.0,
                                min(1.0, max(z_value, 0.0))
                            ),
                            linewidth=0,
                            border=False
                        )
                        viewer.draw_shape(
                            vertices=vertices,
                            position=pos,
                            angle=orientation,
                            filled=True,
                            color=(
                                1.0,
                                0.0,
                                0.0,
                                min(1.0, abs(min(z_value, 0.0)))
                            ),
                            linewidth=0,
                            border=False
                        )

        if self.render_ego_encoding:

            draw_line = merged_centerline.lateral_translate(self.offset_ego)
            if self.grid_size is not None and z.shape[1] > self.grid_size:
                self.dim_reduction = z.shape[1] // self.grid_size
                binned_z = z[:, :(z.shape[1] // self.dim_reduction)*self.dim_reduction].reshape(z.shape[0], z.shape[1] // self.dim_reduction, -1)
                if self.compute_max:
                    z_agg = torch.max(binned_z, 2)[0]
                else:
                    z_agg = torch.mean(binned_z, 2)
            else:
                z_agg = z
            length = draw_line.length/z_agg.shape[1] - 0.2
            width = 2.2
            vertices = np.array([
                [- 0.5 * length, - 0.5 * width], [- 0.5 * length, + 0.5 * width],
                [+ 0.5 * length, + 0.5 * width], [+ 0.5 * length, - 0.5 * width],
                [- 0.5 * length, - 0.5 * width]
            ])

            for t in range(z_agg.shape[1]):
                   
                arclength = draw_line.length*t/(z_agg.shape[1] - 1)
                pos = draw_line(arclength)
                orientation = draw_line.get_direction(arclength)

                if viewer.theme == ColorTheme.DARK:
                    if self.render_lanelet_encodings:
                        z_value = 0.5 + self.multiplier*(z_agg[0, t].item())
                        z_value = np.clip(z_value, 0, 1)
                        color = (
                            z_value,
                            z_value,
                            z_value,
                            self.alpha
                        )
                    elif self.multi_color:
                        z_value = 1.2*z_agg[0, t].item()
                        color = (
                            min(1.0, abs(min(z_value, 0.0))),
                            min(1.0, max(z_value, 0.0)),
                            0.0,
                            1.0
                        )                        
                    else:
                        z_value = 0.5 + self.multiplier*(z_agg[0, t].item())
                        z_value = np.clip(z_value, 0, 1)

                        if z_value < 0.25: # don't draw black circles
                            continue
                        color = (
                            0,
                            z_value,
                            0,
                            self.alpha
                        )

                    viewer.draw_shape(
                        vertices=vertices,
                        position=pos,
                        angle=orientation,
                        filled=True,
                        color=color,
                        linewidth=0,
                        border=False
                    )

                else:
                    z_value = 1.2*z_agg[0, t].item()
                    viewer.draw_shape(
                        vertices=vertices,
                        position=pos,
                        angle=orientation,
                        filled=True,
                        color=(
                            0.0,
                            0.9,
                            0.0,
                            min(1.0, abs(min(z_value, 0.0)))
                        ),
                        linewidth=0,
                        border=False
                    )
                    viewer.draw_shape(
                        vertices=vertices,
                        position=pos,
                        angle=orientation,
                        filled=True,
                        color=(
                            0.0,
                            0.0,
                            0.0,
                            min(1.0, max(z_value, 0.0)),
                        ),
                        linewidth=0,
                        border=False
                    )
        # try:
        #     user_input = input("Please enter lane index to plot: (Enter q to disable)")
        #     if user_input.lower() == 'q':
        #         plots = False
        #         continue_plotting = False
        #         print("Disabled plotting")
        #     else:
        #         lanelet_idx = int(user_input)
        #         continue_plotting = True
        # except:
        #     continue_plotting = False
        #if continue_plotting:qq
        if should_create_plot:
            draw_distrs = self.plot_subcomponents
            if draw_distrs:
                try:
                    occ_prob_components = output_pred[1][1]['occ_prob_components'].squeeze(0)
                except KeyError:
                    occ_prob_components = output_pred[1]['occ_prob_components'].squeeze(0)
                n_distr = occ_prob_components.shape[-1]

            plt.cla()
            fig, ax = plt.subplots(figsize=(8, 8))
            #frames = list(range(time_horizon))
            frame_interval = 2
            legend_interval = 5
            draw_nonlegend = False
            def draw_frame(frame: int): #, joint_occ_probs, occ_probs):
                legend = []
                #for frame in frames:
                ax.clear()
                ax.set_ylim(0, 1)
                ax.set_xlim(0, path_length)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                #ax.get_xaxis().set_major_formatter(FormatStrFormatter(r'%d m'))
                #ax.set_title(f"t+{frame+1}")
                #ax.set_title(f"Joint lanelet occupancy predictions")
                ax.set_xlabel(r"$x \; [m]$")
                ax.set_ylabel(r"$\hat{o}(x, \tau)$")

                c = 0
                for t in range(0, time_horizon, frame_interval):

                    x = np.linspace(0, path_length, resolution)
                    y = joint_occ_probs[0, t, :].detach().numpy()

                    # print(t, color, frame, x.shape, y.shape, y.mean())
                    
                    time = params.scenario.dt * t
                    
                    if draw_distrs and (t == 0 or t == time_horizon - frame_interval):
                        color = (1 - t/time_horizon, 0, t/time_horizon, 0.6)
                        for distr_idx in range(n_distr):
                            y_c = occ_prob_components[0, t, :, distr_idx].detach().numpy()
                            ax.plot(x, y_c, linestyle='dotted', color=color, linewidth=1.0)
                    if c % legend_interval == 0:
                        color = (1 - t/time_horizon, 0, t/time_horizon)
                        ax.plot(x, y, color=color, linewidth=1.4, linestyle='-')
                        legend.append(fr'$\tau = {time:.1f} \; s$')
                    elif draw_nonlegend:
                        color = (1 - t/time_horizon, 0, t/time_horizon, 0.3)
                        ax.plot(x, y, color=color, linewidth=1.0, label='_nolegend_', linestyle='-')
                        
                    c+= 1

                ax.legend(legend)
                return [ax]
                    #
                    #plt.show()
                    #ax.vlines(mu_t[frame, 0, :], ymin=1-0.1*w_arr, ymax=1+0.1*w_arr)
                    #ax.vlines(data.pos.ravel(), ymin=0.0, ymax=0.1, colors='green')

                #animation = FuncAnimation(fig, draw_frame, 20, blit=True)
                #writergif = PillowWriter(fps=len(frames)/7.5)
            ax = draw_frame(0)

            # col_map = LinearSegmentedColormap.from_list('BlackGreen', [(1, 0, 0), (0, 0, 1)], N=2000)
            # c_map_ax = fig.add_axes([1.1, 0.2, 0.02, 0.6])
            # c_map_ax.axes.get_xaxis().set_visible(True)
            # c_map_ax.axes.get_yaxis().set_visible(True)
            # #plt.gca().xaxis.set_major_formatter(StrMethodFormatter(f'%d km'))

            # # and create another colorbar with:
            # matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=col_map, orientation = 'vertical')
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['left'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].grid(axis='y')

            plt.tight_layout()
            
            if self.enable_plots:
                file_name = f"occpred_path_{self.call_count}_avg_{joint_occ_probs.mean().item():.4f}"
                os.makedirs(self.figure_dir, exist_ok=True)
                plt.savefig(os.path.join(self.figure_dir, file_name + '.pdf')) #, facecolor=(0.975, 0.975, 0.975))
                viewer.screenshot(os.path.join(self.figure_dir, file_name + '.png'))
                print(f"Saved figures to {self.figure_dir=}")
            if debugger_is_active() and self.show_plots_debugger:
                plt.show()

            plt.close('all') 
            plt.clf()
            plt.cla()

            grid_height, grid_width = calc_closest_factors(z.shape[1])
            selected_z = z.view(grid_height, grid_width).detach().cpu().numpy()
            plt.grid(None)
            
            plt.axis('off')
            cmap = LinearSegmentedColormap.from_list('BlackGreen', [(0, 0, 0), (0, 1, 0)], N=2000)
            heatmap = plt.imshow(selected_z, cmap=cmap)
            cbar = plt.colorbar(heatmap, orientation="horizontal", pad=0.1)
            # cbar.set_ticks([])
            plt.rcParams["axes.grid"] = False
            if self.enable_plots:
                plt.savefig(os.path.join(self.figure_dir, f'occenc_{self.call_count}.pdf'), transparent=True)
            if debugger_is_active() and self.show_plots_debugger:
                plt.show()

            plt.close('all') 
            plt.clf()
            plt.cla()

            self.plot_count += 1
