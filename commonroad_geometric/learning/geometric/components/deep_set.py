from typing import Literal

import torch
from torch import nn, Tensor


class DeepSetInvariant(nn.Module):
    r"""Invariant deep set model
    https://arxiv.org/pdf/1703.06114.pdf

    :math:`f(X) = \rho \left( \sum_{x \in X} \phi(x) \right)`

    Args:
        element_transform: :math:`\phi`
        output_transform: :math:`\rho`
        aggregation: aggregation function
    """

    def __init__(
        self,
        element_transform: nn.Module,
        output_transform: nn.Module,
        aggregation: Literal["sum", "max", "min", "mean"] = "sum",
    ):
        super().__init__()
        self.element_transform = element_transform
        self.output_transform = output_transform
        self.aggregation = aggregation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self.element_transform, "reset_parameters"):
            self.element_transform.reset_parameters()
        if hasattr(self.output_transform, "reset_parameters"):
            self.output_transform.reset_parameters()

    def forward(self, input_set: Tensor):
        # input size: (..., M, D_in)
        # M is the sequence dimension
        # D_in is the input feature dimension
        input_size = input_set.size()
        D_in = input_size[-1]
        input_set = input_set.view(-1, D_in)
        transformed_set = self.element_transform(input_set)
        transformed_set = transformed_set.view(*input_size[:-1], -1)  # size: (..., M, D_element)

        if self.aggregation == "sum":
            aggregated_set = transformed_set.sum(dim=-2)
        elif self.aggregation == "max":
            aggregated_set, _ = transformed_set.max(dim=-2)
        elif self.aggregation == "min":
            aggregated_set, _ = transformed_set.min(dim=-2)
        elif self.aggregation == "mean":
            aggregated_set = transformed_set.mean(dim=-2)
        else:
            raise ValueError(f"Unknown aggregation value: {self.aggregation}")

        out = self.output_transform(aggregated_set)
        # output size: (..., D_out)
        # D_out is the output feature dimension
        return out


class DeepSetEquivariant(nn.Module):
    # TODO implementation is specialized to the 1-d feature case
    r"""Equivariant deep set model
    https://arxiv.org/pdf/1703.06114.pdf
    """

    def __init__(
        self,
        non_linearity: nn.Module,
        aggregation: Literal["sum", "max"] = "sum",
    ):
        super().__init__()
        self.lmbda = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.aggregation = aggregation
        self.non_linearity = non_linearity

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lmbda.data = torch.tensor([0.5])
        self.gamma.data = torch.tensor([0.5])

    def forward(self, input_set: Tensor):
        if self.aggregation == "sum":
            aggr = input_set.sum(dim=-1, keepdim=True)
        elif self.aggregation == "max":  # aka maxpool
            aggr, _ = input_set.max(dim=-1, keepdim=True)
        else:
            raise TypeError(f"Unknown aggregation {self.aggregation}")
        return self.non_linearity(self.lmbda * input_set + self.gamma * aggr)
