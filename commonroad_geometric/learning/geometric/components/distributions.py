from torch import Tensor
import torch


def uniform_cdf(
    x: Tensor,
    m: Tensor,
    s: Tensor
) -> Tensor:

    a = m - s
    b = m + s
    inside_indeces = (x >= a) & (x <= b)
    above_indeces = x > b
    cdf = inside_indeces.int() * (x - a)/(2*s + 1e-7) + above_indeces.float()
    return cdf


def triangular_cdf(
    x: Tensor,
    m: Tensor,
    s: Tensor
) -> Tensor:

    a = m - s
    b = m + s
    
    left_indeces = (x > a) & (x <= m)
    right_indeces = (x > m) & (x < b)
    above_indeces = x >= b

    cdf = left_indeces.int() * (x - a)**2/((b - a)*(m - a)) + \
          right_indeces.int() * (1 - (b - x)**2/((b - a)*(b - m))) + \
          above_indeces.float()

    return cdf


def laplace_cdf(
    x: Tensor,
    m: Tensor,
    s: Tensor
) -> Tensor:    
    left_indeces = x < m
    right_indeces = x >= m

    cdf = left_indeces.int() * 0.5*torch.exp((x-m)/(s + 1e-5)) + \
          right_indeces.int() * (1 - 0.5*torch.exp((m-x)/(s + 1e-5)))
    return cdf
