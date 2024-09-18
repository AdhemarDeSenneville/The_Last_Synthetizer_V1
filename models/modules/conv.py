# Code from Adhémar de Senneville inspired by https://github.com/archinetai/audio-encoders-pytorch
# Simplified conv Modules

import torch.nn as nn
from torch import Tensor

from .activation import get_activation

def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)


def Downsample1d(in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor,
        padding=factor * (kernel_multiplier // 2),
    )


def Upsample1d(in_channels: int, out_channels: int, factor: int) -> nn.Module:
    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
    return ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * 2,
        stride=factor,
        padding=factor // 2 + factor % 2,
        output_padding=factor % 2,
    )

class ConvBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            num_groups: int = 4,
            use_norm: bool = True,
            activation: str = 'relu',
        ) -> None:
        super().__init__()

        #assert in_channels % num_groups == 0, f"num_channels ({in_channels}) must be divisible by num_groups ({num_groups})"
        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm and in_channels % num_groups == 0
            else nn.Identity()
        )

        self.activation = get_activation(activation, in_features=in_channels)

        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)