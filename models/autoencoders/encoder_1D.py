"""
Code from Adhémar de Senneville

Fully Convolutional Autoencoder

Key Features:
- ResNet Blocks: ResNet 1D Conv Blocks. With Groupnorm + Custom Activation Functions
- LSTM Layers: Optional bidirectional LSTM (Non Streamable) layer befor last convolution for high temporal recepive field.
- Variational AE: Optional Variational mode.
"""

import torch
import torch.nn as nn
from torch import Tensor

from typing import List
from math import prod

from ..modules.lstm import LSTM
from ..modules.conv import ConvBlock1d, Conv1d, Upsample1d, Downsample1d
from .utils import pre_process, post_process

class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        num_groups: int = 4,
        activation: str = 'relu',
    ) -> None:
        super().__init__()

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

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)
    
class Encoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        factors: List[int],
        res_blocks: int = 2,
        activation: str = 'relu',
        use_norm: bool = True,
        num_groups: int = 4,
        variational: bool = True,
        lstm: bool = True,
    ) -> None:
        super().__init__()

        assert len(channels) == len(factors)

        # Encoder path with downsampling
        self.encoder = nn.ModuleList()
        last_channels = in_channels
        for i in range(len(channels)):
            layer = nn.ModuleList()
            for _ in range(res_blocks):
                layer.append(
                    ResnetBlock1d(
                        in_channels=last_channels,
                        out_channels=channels[i],
                        use_norm=use_norm,
                        num_groups=num_groups,
                        activation=activation,
                    )
                )
                last_channels = channels[i]
            #if i < len(channels) - 1:
            layer.append(Downsample1d(in_channels=channels[i], out_channels=channels[i], factor=factors[i]))
            self.encoder.append(layer)
        
        if lstm:
            self.lstm = LSTM(
                dimension = channels[-1],
                num_layers = 1,
                skip = True,
                bidirectional = True,
            )
        else:
            self.lstm = nn.Identity()

        # Final convolution
        self.variational = variational
        if variational:
            self.final_mean = Conv1d(in_channels=channels[-1], out_channels=out_channels, kernel_size=1)
            self.final_logsigma = Conv1d(in_channels=channels[-1], out_channels=out_channels, kernel_size=1)
        else:
            self.final_conv = Conv1d(in_channels=channels[-1], out_channels=out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        for layer in self.encoder:
            for block in layer[:-1]:
                x = block(x)
            x = layer[-1](x)
        
        x = self.lstm(x)

        if self.variational:
            mean = self.final_mean(x)
            log_std = self.final_logsigma(x)

            std = torch.exp(log_std) + 1e-5
            z = torch.randn_like(mean) * std + mean

            return {
                'z': z, 
                'z_mean': mean, 
                'z_log_std': log_std,
            }
        else:
            return {
                'z': self.final_conv(x)
            }
    
class Decoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        factors: List[int],
        res_blocks: int = 2,
        activation: str = 'relu',
        use_norm: bool = True,
        num_groups: int = 4,
        variational: bool = True,
        lstm: bool = True,
    ) -> None:
        super().__init__()

        assert len(channels) == len(factors)

        self.initial_conv = Conv1d(in_channels=out_channels, out_channels=channels[-1], kernel_size=1)

        if lstm:
            self.lstm = LSTM(
                dimension = channels[-1],
                num_layers = 1,
                skip = True,
                bidirectional = True,
            )
        else:
            self.lstm = nn.Identity()

        # Decoder path with upsampling
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            layer = nn.ModuleList()
            layer.append(Upsample1d(in_channels=channels[i], out_channels=channels[i], factor=factors[i]))
            for j in range(res_blocks):

                output_channels = channels[i] if j != res_blocks - 1 else channels[max(i-1, 0)]
                layer.append(
                    ResnetBlock1d(
                        in_channels=channels[i],
                        out_channels=output_channels,
                        use_norm=use_norm,
                        num_groups=num_groups,
                        activation=activation,
                    )
                )

            self.decoder.append(layer)

        # Final convolution
        self.final_conv = Conv1d(in_channels=channels[0], out_channels=in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.initial_conv(x)

        x = self.lstm(x)
        # Encoder
        for layer in self.decoder:
            for block in layer[:-1]:
                x = block(x)
            x = layer[-1](x)

        x = self.final_conv(x)

        return {'x_hat': x}


class Autoencoder1d(nn.Module):
    def __init__(
        self, **nn_cfg
    ) -> None:
        super().__init__()
        
        self.encoder = Encoder1d(**nn_cfg)
        self.decoder = Decoder1d(**nn_cfg)

        self.eff_padding = int(prod(nn_cfg['factors']))

    def forward(self, x):
        x, x_length = pre_process(x, self.eff_padding)

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['z'])

        decoder_output['x_hat'] = post_process(decoder_output['x_hat'],x_length)

        return {
            **encoder_output,
            **decoder_output,
        }

if __name__ == '__main__':
    
    config = {
        'in_channels': 1,
        'out_channels': 16,
        'channels': [16, 32, 64, 64],
        'factors': [4, 4, 4, 4],
        'res_blocks': 2,
        'activation': 'Snake',
        'use_norm': True,
        'num_groups': 4,
        'variational': True,
        'lstm': True,
    }
    # Test models
    test_encoder_model = Encoder1d(**config)
    test_decoder_model = Decoder1d(**config)
    test_autoencoder_model = Autoencoder1d(**config)

    # Example input
    input_tensor = torch.randn(7, 1, 1025)

    # Test encoder
    z_tensor = test_encoder_model(input_tensor)
    print("z     shape",z_tensor['z'].shape)

    # Test decoder
    x_tensor = test_decoder_model(z_tensor['z'])
    print("x_hat shape",x_tensor['x_hat'].shape)

    # Test auto-encoder
    x_tensor = test_autoencoder_model(input_tensor)
    print("x_hat shape",x_tensor['x_hat'].shape)
