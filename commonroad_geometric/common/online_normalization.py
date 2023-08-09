# from typing import Tuple
# import torch

# from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin


# # TODO: Support numpy backend
# class OnlineNormalizer(AutoReprMixin):
#     def __init__(self, clip: float = 10.0, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), update_threshold: int = 1e4):
#         """
#         Calulates the running mean and std of a data stream.

#         :param clip: clips values outside this range
#         :param epsilon: helps with arithmetic issues
#         :param shape: the shape of the data stream's output
#         """
#         self._clip = clip
#         self._mean = torch.zeros(shape, dtype=torch.float32)
#         self._sq_differences = torch.zeros(shape, dtype=torch.float64)
#         self._var = torch.zeros(shape, dtype=torch.float32)
#         self._episilon = epsilon
#         self._count = epsilon
#         self._update_threshold = update_threshold

#     def _update(self, arr: torch.Tensor) -> None:
#         arr_finite = arr[torch.isfinite(arr).any(dim=1)] # TODO check column-wise
#         batch_count = arr_finite.shape[0]
#         if batch_count == 0:
#             return
#         new_count = self._count + batch_count

#         if self._count == 1:
#             batch_mean = torch.mean(arr_finite, axis=0)
#             self._mean = batch_mean
#         else:
#             delta = arr_finite - self._mean
#             self._mean += delta.sum(axis=0) / new_count
#             new_delta = arr_finite - self._mean
#             self._sq_differences += (delta * new_delta).sum(axis=0)
#             self._var = self._sq_differences / (new_count - 1)

#         self._count = new_count

#     @property
#     def mean(self) -> torch.Tensor:
#         return self._mean

#     @property
#     def var(self) -> torch.Tensor:
#         return self._var

#     @property
#     def std(self) -> torch.Tensor:
#         return torch.sqrt(self._var + self._episilon)

#     @property
#     def count(self) -> int:
#         return int(self._count)

#     def normalize(self, arr: torch.Tensor, update: bool = True) -> torch.Tensor:
#         if not isinstance(arr, torch.Tensor):
#             arr = torch.tensor([arr], dtype=torch.float32)
#         if update and self._count < self._update_threshold:
#             self._update(arr)
#         val = torch.clip((arr - self.mean) / self.std, -self._clip, self._clip)
#         return val

#     def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
#         if not isinstance(arr, torch.Tensor):
#             arr = torch.tensor([arr], dtype=torch.float32)
#         val = (arr * torch.sqrt(self._var + self._episilon)) + self._mean
#         return val
