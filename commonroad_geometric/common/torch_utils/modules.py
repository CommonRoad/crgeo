from torch import nn


def num_parameters(model: nn.Module) -> int:
    return sum(
        parameter.size().numel()
        for parameter in model.parameters()
    )


def num_bytes(model: nn.Module) -> int:
    return sum(
        parameter.size().numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def freeze_weights(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False
