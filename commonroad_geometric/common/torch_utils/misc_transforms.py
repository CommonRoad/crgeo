from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def greedy_argmin_permutation(z: Tensor) -> Tensor:
    # TODO: Documentation
    # TODO: cleanup
    mask = torch.zeros_like(z, device=z.device, dtype=torch.bool)
    a = z

    dim1: int = -2
    dim2: int = -1

    asort, aidx = torch.sort(a, dim=dim2, stable=True)
    row_order = torch.sort(asort[..., 0], dim=dim2, stable=True)[1]
    row_order_view = row_order.unsqueeze(-1).repeat(*(row_order.ndim*[1] + [z.shape[dim2]]))
    b = torch.gather(a, dim1, row_order_view)
    # bsort = asort[row_order]
    #bidx = torch.gather(aidx, dim1, row_order_view)

    #choices = z.new_zeros((*list(z.shape)[:dim1], z.shape[dim2], z.shape[dim2]), dtype=torch.bool)
    choices = z.new_zeros((*list(z.shape)[:dim1], z.shape[dim2]), dtype=torch.long)
    mask = z.new_zeros((*list(z.shape)[:dim1], z.shape[dim2]), dtype=torch.bool)

    for r in range(z.shape[dim2]):
        b_row = b[:, r, :]
        c = b_row + torch.zeros_like(b_row).masked_fill(mask, np.inf)
        c_choices = torch.argmin(c, dim=1)
        c_mask = z.new_zeros((*list(z.shape)[:dim1], z.shape[dim2]), dtype=torch.bool).scatter(
            1, c_choices.unsqueeze(-1), 1
        )
        choices[:, r] = c_choices
        mask.bitwise_or_(c_mask)

    a_choices = torch.empty_like(choices).scatter(1, row_order, choices)
    return a_choices


def decompose_dir_magn(z: Tensor, start_dim=1) -> Tuple[Tensor, Tensor]:
    # TODO: Documentation
    lower = torch.tril(z, diagonal=-1)
    upper = torch.transpose(torch.triu(z, diagonal=1), start_dim, start_dim+1)

    direction = lower - upper
    magnitude = lower**2 + upper**2

    bi_direction = direction - torch.transpose(direction, start_dim, start_dim+1)
    bi_magnitude = magnitude + torch.transpose(magnitude, start_dim, start_dim+1)

    return bi_direction, bi_magnitude


def decompose_sum(z: Tensor, start_dim=1, power: int = 1) -> Tensor:
    # TODO: Documentation
    lower = torch.tril(z, diagonal=-1)
    upper = torch.transpose(torch.triu(z, diagonal=1), start_dim, start_dim+1)

    sum = lower**power + upper**power

    bi_sum = sum + torch.transpose(sum, start_dim, start_dim+1)

    return bi_sum


def decompose_signed_max(z: Tensor, start_dim=1) -> Tensor:
    # TODO: Documentation
    lower = torch.tril(z, diagonal=-1)
    upper = torch.transpose(torch.triu(z, diagonal=1), start_dim, start_dim+1)

    concat = torch.stack([
        lower,
        upper
    ], dim=0)

    z = signed_max(concat, dim=0)

    y = z + torch.transpose(z, start_dim, start_dim+1)

    return y


def signed_max(z: Tensor, dim: int = 0) -> Tensor:
    # TODO: Documentation
    max_ = torch.amax(z, dim=dim)
    min_ = torch.amin(z, dim=dim)
    max_greater = (torch.abs(max_) >= torch.abs(min_)).int()
    signed_max = max_greater*max_ + (1 - max_greater)*min_
    return signed_max
