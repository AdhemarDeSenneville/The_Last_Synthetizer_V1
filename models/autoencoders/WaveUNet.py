# Code from Adhémar de Senneville
# Fully conv Unet inpired from Diff A Riff

import torch
import torch.nn as nn
from torch import Tensor

from typing import List
from math import prod

from ..modules.conv import ConvBlock1d, Conv1d, Upsample1d, Downsample1d
from ..conditionning.utils import get_timestep_embedding
from .utils import pre_process, post_process


class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        num_groups: int = 1,
        activation: str = 'relu',
    ) -> None:
        super().__init__()

        self.dense = nn.Linear(
            in_features=t_channels,
            out_features=out_channels,
        )

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
            activation=activation,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
            activation=activation,
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.block1(x)
        t = self.dense(t)
        h = h + t.unsqueeze(-1)
        h = self.block2(h)
        return h + self.to_out(x)

class UNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        factors: List[int],
        t_channels: int,
        t_factor: int,
        res_blocks: int = 2,
        activation: str = 'relu',
        use_norm: bool = True,
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        
        # Encode step embedding
        self.t_channels_in = t_channels
        t_channels_out = t_factor * t_channels
        self.dense_t = nn.Sequential(
            nn.Linear(t_channels    , t_channels_out),
            nn.ReLU(),
            nn.Linear(t_channels_out, t_channels_out),
            nn.ReLU()
        )

        # Encoder path with downsampling
        self.encoder = nn.ModuleList()
        for i in range(len(channels)):
            layer = nn.ModuleList()
            for _ in range(res_blocks):
                layer.append(
                    ResnetBlock1d(
                        in_channels=in_channels,
                        out_channels=channels[i],
                        t_channels=t_channels_out,
                        use_norm=use_norm,
                        num_groups=num_groups,
                        activation=activation,
                    )
                )
                in_channels = channels[i]

            layer.append(Downsample1d(in_channels=channels[i], out_channels=channels[i], factor=factors[i]))
            self.encoder.append(layer)
        
        self.bottle_neck = ResnetBlock1d(
                        in_channels=channels[-1],
                        out_channels=channels[-1],
                        t_channels=t_channels_out,
                        use_norm=use_norm,
                        num_groups=num_groups,
                        activation=activation,
                    )

        # Decoder path with upsampling
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            layer = nn.ModuleList()
            layer.append(Upsample1d(in_channels=channels[i], out_channels=channels[i], factor=factors[i]))
            in_channels = 2*channels[i]
            for j in range(res_blocks):

                output_channels = channels[i] if j != res_blocks - 1 else channels[max(i-1, 0)]
                layer.append(
                    ResnetBlock1d(
                        in_channels=in_channels,
                        out_channels=output_channels,
                        t_channels=t_channels_out,
                        use_norm=use_norm,
                        num_groups=num_groups,
                        activation=activation,
                    )
                )
                in_channels = channels[i]

            self.decoder.append(layer)

        # Final convolution
        self.final_conv = Conv1d(in_channels=channels[0], out_channels=out_channels, kernel_size=1)

        self.eff_padding = int(prod(factors))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:

        x, x_length = pre_process(x, self.eff_padding)
        
        # Encode time
        t = get_timestep_embedding(t, self.t_channels_in)
        t = self.dense_t(t)

        # Encoder # Unoptimized, for debugging
        skips = []
        for layer in self.encoder:
            for block in layer[:-1]:
                x = block(x,t)
            skips.append(x)
            x = layer[-1](x)
            
        x = self.bottle_neck(x,t)

        # Decoder
        skips = skips[::-1]
        for i, layer in enumerate(self.decoder):
            x = layer[0](x)
            x = torch.cat((x, skips[i]), dim=-2)
            for block in layer[1:]:
                x = block(x,t)

        x = self.final_conv(x)
        x = post_process(x, x_length)

        return x
    

if __name__ == '__main__':
    unet_model = UNet1d(
        in_channels=16,
        out_channels=16,
        t_channels=64,
        channels=[32, 64, 128, 256],
        factors=[2, 2, 2, 2],
        t_factor=2,
        res_blocks=2,
    )

    # Example input tensor
    x_tensor = torch.randn(7, 16, 1025)
    t_tensor = torch.randn(7)

    # Forward passl
    output_tensor = unet_model(x_tensor, t_tensor)

    # Print
    print(f"{x_tensor.shape = }")
    print(f"{t_tensor.shape = }")
    print(f"{output_tensor.shape = }")
