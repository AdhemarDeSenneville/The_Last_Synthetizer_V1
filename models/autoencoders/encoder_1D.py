from math import floor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops import pack, rearrange, reduce, unpack
#from einops_exts import rearrange_many
from torch import Tensor
from torchaudio import transforms

import typing as tp
from math import prod
from torch import nn

# Simplified LSTM Module
class LSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool = True):
        super().__init__()
        self.skip = skip
        self.bidirectional = bidirectional
        self.dimention = dimension

        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional = bidirectional)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)

        if self.bidirectional:
            y = y[...,-self.dimention:]

        if self.skip:
            y = y + x

        y = y.permute(1, 2, 0)
        return y

# Simplified conv Modules
def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)


def Downsample1d(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
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
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        num_groups: int = 1,
        use_norm: bool = True,
        activation: tp.Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels) # WARNING
            if use_norm
            else nn.Identity()
        )
        self.activation = activation()
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
        activation: tp.Type[nn.Module] = nn.ReLU,
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
            activation=activation,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
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
        activation: tp.Type[nn.Module] = nn.ReLU,
        use_norm: bool = True,
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
        activation: tp.Type[nn.Module] = nn.ReLU,
        use_norm: bool = True,
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
    
    def pre_process(self, x):
        """
        Zero-pad the temporal dimension (dimension 2) to make its length a multiple of eff_padding.
        """
        original_length = x.shape[2]
        pad_length = (self.eff_padding - (original_length % self.eff_padding)) % self.eff_padding

        if pad_length > 0:
            x = F.pad(x, (0, pad_length))

        return x, original_length

    def post_process(self, x, original_length):
        """
        Remove any excess padding in the temporal dimension to restore the original length.
        """
        return x[:, :, :original_length]

    def forward(self, x):
        x, x_length = self.pre_process(x)

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['z'])

        decoder_output['x_hat'] = self.post_process(decoder_output['x_hat'],x_length)

        return {
            **encoder_output,
            **decoder_output,
        }

if __name__ == '__main__':
    
    config = {
        'in_channels': 2,
        'out_channels': 16,
        'channels': [16, 32, 64, 64],
        'factors': [4, 4, 4, 4],
        'res_blocks': 2,
        'activation': nn.ReLU,
        'use_norm': True,
        'variational': True,
        'lstm': True,
    }
    # Test models
    test_encoder_model = Encoder1d(**config)
    test_decoder_model = Decoder1d(**config)
    test_autoencoder_model = Autoencoder1d(**config)

    # Example input
    input_tensor = torch.randn(7, 2, 1025)

    # Test encoder
    z_tensor = test_encoder_model(input_tensor)
    print(z_tensor['z'].shape) 

    # Test decoder
    x_tensor = test_decoder_model(z_tensor['z'])
    print(x_tensor['x_hat'].shape) 

    # Test auto-encoder
    x_tensor = test_autoencoder_model(input_tensor)
    print(x_tensor['x_hat'].shape) 
