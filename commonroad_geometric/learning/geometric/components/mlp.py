import sys
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, ReLU, Sequential as Seq
from torch.nn.modules.batchnorm import BatchNorm1d

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class MLP(torch.nn.Module):
    """Multilayer perceptron (MLP) module."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        net_arch: Optional[Union[int, Sequence[int]]] = None,
        norm_method: Literal['layer', 'batch', 'none'] = 'none',
        dropout: float = 0.0,
        highway: bool = False,
        activation_cls: Any = ReLU,
        bias_in: bool = True,
        bias_hidden: bool = True,
        bias_out: bool = True,
        norm_momentum: float = 0.1
    ):
        super(MLP, self).__init__()

        if net_arch is None:
            net_arch = []
        elif isinstance(net_arch, int):
            net_arch = [net_arch]
        net_arch += [output_size]

        if not isinstance(activation_cls, list):
            activation_cls = [activation_cls] * (len(net_arch))

        self.highway = highway

        self.norm_method = norm_method
        if self.highway:
            self.linear = Linear(input_size, output_size, bias=bias_in)
            self.gate = Linear(input_size, output_size, bias=bias_in)
        else:
            self.linear = None
            self.gate = None

        seq = []

        seq.extend([
            Linear(input_size, net_arch[0], bias=bias_in),
        ])

        if norm_method == 'layer':
            seq.append(LayerNorm(net_arch[0]))
        elif norm_method == 'batch':
            seq.append(BatchNorm1d(net_arch[0], momentum=norm_momentum))
        seq.extend([activation_cls[0]()])
        if dropout > 0:
            seq.append(Dropout(p=dropout))

        for h_idx in range(len(net_arch) - 1):
            h = net_arch[h_idx]
            if norm_method == 'batch' and h_idx < len(net_arch) - 2:
                seq.extend([Linear(h, net_arch[h_idx + 1], bias=False)])
            elif h_idx < len(net_arch) - 2:
                seq.extend([Linear(h, net_arch[h_idx + 1], bias=bias_hidden)])
            else:
                seq.extend([Linear(h, net_arch[h_idx + 1], bias=bias_out)])

            if h_idx < len(net_arch) - 2:
                if norm_method == 'layer':
                    seq.append(LayerNorm(net_arch[h_idx + 1]))
                elif norm_method == 'batch':
                    seq.append(BatchNorm1d(net_arch[h_idx + 1], momentum=norm_momentum))
                seq.extend([activation_cls[h_idx + 1]()])
                if dropout > 0:
                    seq.append(Dropout(p=dropout))

        self.network = Seq(*seq)

    def forward(self, x: Tensor) -> Tensor:
        z = self.network.forward(x)
        if self.highway:
            gate = torch.sigmoid(self.gate(x))
            linear = self.linear(x)

            y = gate * z + (1 - gate) * linear
        else:
            y = z
        return y
