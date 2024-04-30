from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Tuple, TypeVar, Union, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.optim.optimizer import Optimizer
from torch_geometric.data import HeteroData
from torch_geometric.data.data import BaseData

if TYPE_CHECKING:
    from commonroad_geometric.dataset.commonroad_data import CommonRoadData


T_Data = TypeVar("T_Data", bound=BaseData)


def to_scalar(x: Any) -> Union[float, None]:
    """Type-agnostic scalar conversion."""
    if isinstance(x, (Tensor, np.ndarray)) and x.ndim == 0:
        return x.item()
    try:
        return float(x)
    except ValueError:
        return None


def is_scalar(x: Any) -> bool:
    """Checks if value is scalar."""
    if isinstance(x, (Tensor, np.ndarray)) and x.ndim == 0:
        return True
    try:
        float(x)
        return True
    except ValueError:
        return False


def is_finite(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return cast(bool, torch.isfinite(x).all().item())
    elif isinstance(x, np.ndarray):
        return np.isfinite(x).all()
    return math.isfinite(x)


def to_float32(data: T_Data) -> T_Data:
    """Converts all attributes of PyTorch Geometric graph instance to float32 dtype."""
    for k in keys(data):
        if hasattr(data[k], 'dtype') and data[k].dtype == torch.float64:
            data[k] = data[k].to(torch.float32)
    return data

def keys(data: T_Data):
    if isinstance(data.keys, list):
        return data.keys # older pygeo
    return data.keys() # newer pygeo

def to_float32_tensors(ndarrays: List[np.ndarray]) -> List[Tensor]:
    return [
        torch.from_numpy(arr).type(torch.float32)
        for arr in ndarrays
    ]


def to_padded_sequence(ndarrays: List[np.ndarray]) -> Tensor:
    return pad_sequence(
        sequences=to_float32_tensors(ndarrays),
        batch_first=True,
        padding_value=0.0,
    )


def optimizer_to(optim: Optimizer, device: torch.device) -> None:
    # https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


@torch.jit.script
def get_index_mapping_by_first_occurrence(arr: torch.Tensor):
    index_dict: Dict[float, int] = {}
    num_unique: int = 0
    indices = torch.empty_like(arr)
    for i, x in enumerate(arr):
        x_item = x.item()
        if x_item not in index_dict:
            index_dict[x_item] = num_unique
            num_unique += 1
        indices[i] = index_dict[x_item]
    return indices


class FlattenDataInsufficientPaddingSize(ValueError):
    pass


def flatten_data(
    data: CommonRoadData,
    padding: int,
    validate: bool = False,
    keys: Optional[Set[str]] = None,
    ignore_keys: Optional[Set[str]] = None,
) -> Dict[str, Tensor]:
    # TODO: Make implementation readable
    # TODO: Unit test
    # TODO: Optimize
    """

    Converts Pytorch Geometric data instance to dictionary of fixed-sized tensors.
    Needed for StableBaselines compatability.

    Args:
        data (CommonRoadData): Input data.
        padding (int): Size (width) of output tensors.
        validate (bool, optional): Whether to reconstruct the input data afterwards for validation. Defaults to False.
        keys (Set[str], optional): Optional set of keys to include in the output. Defaults to None (i.e. all keys).

    Returns:
        Dict[str, Tensor]: Dictionary of fixed-size PyTorch tensors.
    """

    data_dict = data.to_dict()
    tensors: Dict[str, Tensor] = {}
    for k_str, v in data_dict.items():
        if not '-' in k_str:
            continue
        if ignore_keys is not None and k_str in ignore_keys:
            continue
        if isinstance(v, Tensor):
            assert v.ndim == 2, f"{k_str} is not two-dimensionsal: {v.shape}"
            if keys is not None and k_str not in keys:
                continue

            tensors[k_str] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if not isinstance(v2, Tensor):
                    continue
                assert v2.ndim == 2
                full_key = f"{k_str}-{k2}"
                if keys is not None and full_key not in keys:
                    continue
                tensors[full_key] = v2

    out: Dict[str, Tensor] = {}

    for k, size in data.store_sizes.items():
        if size is not None:
            out['_' + k] = torch.tensor([size], dtype=torch.long)

    for k_str, v in tensors.items():
        k = tuple(k_str.split('-'))
        if v.ndim == 1:
            v = v.unsqueeze(1)
        if k[-1] == 'edge_index':
            if not v.shape[1] <= padding:
                raise FlattenDataInsufficientPaddingSize(
                    f"padding was exceeded for {k_str}: ({v.shape[1]} > {padding}), please increase in RLEnvironmentOptions")
            v_padded = pad(v, (0, padding - v.shape[1], 0, 0))
            assert v_padded.shape[1] == padding
        else:
            if not v.shape[0] <= padding:
                raise FlattenDataInsufficientPaddingSize(
                    f"padding was exceeded for {k_str}: ({v.shape[0]} > {padding}), please increase in RLEnvironmentOptions")
            if v.shape[1] == 0:
                # skipping empty feature tensors
                continue
            else:
                v_padded = pad(v, (0, 0, 0, padding - v.shape[0]))
            assert v_padded.shape[0] == padding
        out[k_str] = v_padded

    for k, v in data._global_store.items():
        if ignore_keys is not None and k in ignore_keys:
            continue
        if keys is not None and k not in keys:
            continue
        if isinstance(v, Tensor):
            out[k] = v

    if validate:
        reconstructed_data = reconstruct_data(out)
        reconstructed_dict = reconstructed_data.to_dict()
        data_dict = data.to_dict()
        for k, v in data_dict.items():
            if ignore_keys is not None and k in ignore_keys:
                continue
            if keys is not None and k not in keys:
                continue
            if not isinstance(v, Tensor):
                continue
            if len(v.shape) == 2 and v.shape[1] == 0:
                continue
            if '-' not in k:
                vr = reconstructed_dict[k]
            else:
                k1, k2 = k.rsplit('-', 1)
                if '-' in k1:
                    k1 = tuple(k1.split('-'))
                if k2 not in reconstructed_dict[k1]:
                    continue
                vr = reconstructed_dict[k1][k2]
            assert vr.shape == v.shape
            assert torch.equal(torch.nan_to_num(vr), torch.nan_to_num(v))

        if keys is not None:
            included_keys = set(out.keys())
            assert keys == included_keys, f"keys != set(out.keys()), missing={keys - included_keys}, extra={included_keys - keys}"
    return out


def reconstruct_data(
    tensors: Dict[str, Tensor]
) -> HeteroData:
    # TODO: Make implementation readable
    # TODO: Unit test
    """
    Reconstructs the dictionary output of flatten_data.

    Args:
        tensors (Dict[str, Tensor]): Dictionary of tensors to be converted to HeteroData instance.

    Returns:
        HeteroData: Reconstructed graph instance.
    """
    reconstructed_dict = {}
    global_dict = {}

    batch_masks_dict = {}
    batch_widths_dict = {}

    # inserting node features
    for k_str, v in tensors.items():
        if k_str.startswith('_'):
            continue
        k = tuple(k_str.split('-'))
        k0 = k[:-1]
        key = k0[0] if len(k0) == 1 else k0
        k0_str = '-'.join(k0)
        if len(k0) > 1:
            continue
        if not isinstance(v, Tensor):
            continue
        if k[-1] == 'edge_index':
            continue
        if f'_{k0_str}' not in tensors:
            global_dict[k_str] = v
            continue

        if v.ndim == 2:
            v = v.unsqueeze(0)

        if key not in reconstructed_dict:
            reconstructed_dict[key] = {}
        if key not in batch_widths_dict:
            batch_widths = tensors[f'_{k0_str}']
            batch_size = v.shape[0]
            width = v.shape[1]
            full_batch_matrix = torch.arange(batch_size, device=v.device).unsqueeze(-1).repeat(1, width)
            full_int_idx_matrix = torch.arange(width, device=v.device).unsqueeze(0).repeat(batch_size, 1)
            batch_masks = full_int_idx_matrix < batch_widths
            batch = full_batch_matrix[batch_masks]

            reconstructed_dict[key]['batch'] = batch
            batch_masks_dict[key] = batch_masks
            batch_widths_dict[key] = batch_widths

        batch_values = v[batch_masks_dict[key]]
        reconstructed_dict[key][k[-1]] = batch_values

    # inserting edges
    for k_str, v in tensors.items():
        if k_str.startswith('_'):
            continue
        k = tuple(k_str.split('-'))
        k0 = k[:-1]
        key = k0[0] if len(k0) == 1 else k0
        k0_str = '-'.join(k0)
        if f'_{k0_str}' not in tensors:
            continue
        if len(k0) == 1:
            continue
        if v.ndim == 2:
            v = v.unsqueeze(0)

        if k[-1] == 'edge_index':
            v = v.transpose(1, 2).long()
        if key not in reconstructed_dict:
            reconstructed_dict[key] = {}

        if key not in batch_widths_dict:
            batch_widths = tensors[f'_{k0_str}']
            batch_size = v.shape[0]
            width = v.shape[1]
            full_batch_matrix = torch.arange(batch_size, dtype=torch.long,
                                             device=v.device).unsqueeze(-1).repeat(1, width)
            full_int_idx_matrix = torch.arange(
                width,
                dtype=torch.long,
                device=v.device).unsqueeze(0).repeat(
                batch_size,
                1)
            batch_masks = full_int_idx_matrix < batch_widths
            batch = full_batch_matrix[batch_masks]
            batch_masks_dict[key] = batch_masks
            batch_widths_dict[key] = batch_widths

        if k[-1] == 'edge_index':
            k_from = k[0]
            k_to = k[-2]
            from_widths = batch_widths_dict[k_from]
            to_widths = batch_widths_dict[k_to]
            if from_widths.ndim == 1:
                from_widths = from_widths.unsqueeze(1)
            if to_widths.ndim == 1:
                to_widths = to_widths.unsqueeze(1)
            from_cumsum = torch.cat([torch.tensor([0.0], device=v.device).unsqueeze(1),
                                    torch.cumsum(from_widths, 0)])[:-1]
            to_cumsum = torch.cat([torch.tensor([0.0], device=v.device).unsqueeze(1), torch.cumsum(to_widths, 0)])[:-1]

            from_indices = v[:, :, 0] + from_cumsum
            to_indices = v[:, :, 1] + to_cumsum
            batch_idx_full = torch.stack([from_indices, to_indices], dim=-1)  # TODO optimize
            value_select = batch_idx_full[batch_masks_dict[key]].transpose(0, 1).long()
        else:
            value_select = v[batch_masks_dict[key]]
        reconstructed_dict[key][k[-1]] = value_select

    data = HeteroData(reconstructed_dict)

    for k, v in global_dict.items():
        data[k] = v

    return data


class TensorSizeMismatch(AssertionError):
    pass


def assert_size(t: Tensor, size: Tuple[Union[int, None], ...]) -> None:
    if t.ndim != len(size):
        raise TensorSizeMismatch(f"Expected {len(size)} dimensions but got {t.ndim}")
    t_size = t.size()
    for i, s in enumerate(size):
        if s is not None and t_size[i] != s:
            raise TensorSizeMismatch(f"Expected dimension {i} to be of size {s} but got {t_size[i]}")


def reset_module(value: Any) -> None:
    # copied from torch_geometric/nn/inits.py
    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    elif hasattr(value, "children"):
        for child in value.children():
            reset_module(child)
