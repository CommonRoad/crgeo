import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from commonroad_geometric.common.torch_utils.helpers import assert_size


class UpsampleConv(nn.Module):

    def __init__(self, out_size: int, in_channels: int, out_channels: int):
        super().__init__()
        self.out_size = out_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=5, stride=1, padding=2,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # size: N x C x H x W
        assert_size(x, (None, self.in_channels, None, None))
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="nearest")
        x = self.conv(x)  # TODO residual connection?
        assert_size(x, (None, self.out_channels, self.out_size, self.out_size))
        x = self.norm(x)
        return F.relu(x)
