from torch import Tensor
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import torch

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.torch_utils.pygeo import get_batch_sizes, get_batch_start_indices


def preprocess_conditioning(
    data: CommonRoadData,
    walks: Tensor,
    path_length: float,
    walk_start_length: Optional[Tensor] = None,
    walk_velocity: Optional[Tensor] = None,
    walk_masks: Optional[Tensor] = None,
    ignore_assertion_errors: bool = False
) -> None:
    # TODO the encoder should do this
    # walks
    # N_WALKS x WALK_LENGTH

    # walk_start_length
    # (N_WALKS,)

    walks = walks.long()
    if walks.ndim == 1:
        walks = walks.unsqueeze(0)

    # batch_size = int(data.v.batch.max().item()) + 1
    device = data.device

    walk_lanelet_lengths = data.l.length[walks].squeeze(-1)
    if walk_masks is not None:
        if walk_masks.ndim == 3:
            walk_masks = walk_masks.squeeze(-1)  # TODO ?????????????
        if walk_masks.ndim > 1 and walk_masks.shape[1] == walk_lanelet_lengths.shape[0] and walk_masks.shape[0] == walk_lanelet_lengths.shape[1]:
            walk_masks = walk_masks.permute(1, 0)
            # TODO: WHAT???
        walk_lanelet_lengths *= walk_masks

    if walk_start_length is None:
        walk_total_length = walk_lanelet_lengths.sum(dim=1)
        walk_start_length = (walk_total_length - path_length) * torch.rand(walk_total_length.shape, device=device)

    # if walk_velocity is None:
    #     v_batch_sizes = get_batch_sizes(data.v.batch)
    #     v_selections = (torch.rand(batch_size, device=device) * v_batch_sizes).int()
    #     v_batch_start_indices = get_batch_start_indices(v_batch_sizes, is_batch_sizes=True)
    #     v_batch_selections = v_batch_start_indices + v_selections
    #     walk_velocity = data.v.velocity[v_batch_selections]
    #     data.walk_velocity_flattened = walk_velocity.flatten()
    # rw_selections = (torch.rand(batch_size, device=device) * rw_batch_sizes).int()
    # walk_end_length = walk_start_length + self.config.path_length
    # walk_prior_cum_length = torch.cumsum(torch.cat(
    #     [walk_lanelet_lengths.new_zeros((walk_lanelet_lengths.shape[0], 1)),
    #     walk_lanelet_lengths
    # ], dim=-1), dim=1)[:, :-1]
    walk_post_cum_length = torch.cumsum(walk_lanelet_lengths, dim=1)
    walk_post_cum_length_offset = walk_post_cum_length - walk_start_length.unsqueeze(-1)
    walk_prior_cum_length_offset = torch.clamp(torch.cat(
        [walk_post_cum_length_offset.new_zeros((walk_post_cum_length_offset.shape[0], 1)),
         walk_post_cum_length_offset
         ], dim=-1)[:, :-1], min=0.0)
    walk_prior_cum_length_offset_rel = walk_prior_cum_length_offset / path_length

    walk_start_mask = walk_post_cum_length_offset > 0
    walk_end_mask = walk_prior_cum_length_offset < path_length

    walk_mask = walk_start_mask & walk_end_mask
    if walk_masks is not None:
        walk_mask = walk_mask & walk_masks

    lower_limits = torch.clamp(walk_lanelet_lengths - torch.clamp(walk_post_cum_length_offset, min=0.0), min=0.0)
    upper_limits = torch.clamp(
        walk_lanelet_lengths -
        torch.clamp(
            walk_post_cum_length_offset -
            path_length,
            min=0.0),
        min=0.0)
    lower_limits_rel = lower_limits / walk_lanelet_lengths
    upper_limits_rel = upper_limits / walk_lanelet_lengths

    # assert (walk_total_length > self.config.path_length).all()

    # lower_limits = torch.clamp((lower_bounds.unsqueeze(-1) - walk_indeces), 0, 1)
    # upper_limits = torch.clamp((upper_bounds.unsqueeze(-1) - walk_indeces), 0, 1)
    # limit_intervals = upper_limits - lower_limits
    # cumulative_prior_length = torch.cat([limit_intervals.new_zeros((limit_intervals.shape[0], 1)), limit_intervals], dim=1).cumsum(1)[:, :-1]
    # integration_lengths = upper_limits - lower_limits
    if not ignore_assertion_errors:
        assert (walk_post_cum_length[:, -1] > path_length).all()

    walk_mask_flattened = walk_mask.flatten()
    flattened_walks = walks.flatten()[walk_mask_flattened]
    cumulative_prior_length_flattened = walk_prior_cum_length_offset_rel.flatten()[walk_mask_flattened]
    integration_lower_limits_flattened = lower_limits_rel.flatten()[walk_mask_flattened]
    integration_upper_limits_flattened = upper_limits_rel.flatten()[walk_mask_flattened]
    cumulative_prior_length_flattened_abs = walk_prior_cum_length_offset.flatten()[walk_mask_flattened]
    integration_lower_limits_flattened_abs = lower_limits.flatten()[walk_mask_flattened]
    integration_upper_limits_flattened_abs = upper_limits.flatten()[walk_mask_flattened]
    # integration_lengths_flattened = integration_lengths.flatten()[walk_mask_flattened]

    # data.walks_batch = walks
    # data.walk_masks_batch = walk_masks
    # data.cumulative_prior_length_batch = cumulative_prior_length
    # data.integration_lower_limits_batch = lower_limits
    # data.integration_upper_limits_batch = upper_limits
    # data.integration_lengths_batch = integration_lengths

    data.path_length = path_length
    data.walks = flattened_walks
    data.walk_masks = walk_mask_flattened
    data.cumulative_prior_length = cumulative_prior_length_flattened
    data.integration_lower_limits = integration_lower_limits_flattened
    data.integration_upper_limits = integration_upper_limits_flattened
    data.cumulative_prior_length_abs = cumulative_prior_length_flattened_abs
    data.integration_lower_limits_abs = integration_lower_limits_flattened_abs
    data.integration_upper_limits_abs = integration_upper_limits_flattened_abs
