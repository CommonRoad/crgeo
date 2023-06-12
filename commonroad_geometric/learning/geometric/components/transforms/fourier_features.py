import torch

class FourierFeatures(torch.nn.Module):
    def __init__(self, num_input_channels: int, mapping_size: int, scale: float = 1.0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size//2)) * scale
        self._c = torch.randn((mapping_size//2,)) * scale

    def forward(self, x):
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())

        batch_size, channels = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        z = x @ self._B.to(x.device) + self._c.to(x.device)
        y = torch.cat([torch.sin(z), 1 - torch.cos(z)], dim=1)
        return y
