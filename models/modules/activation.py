# Code from Adhémar de Senneville and https://github.com/NVIDIA/BigVGAN/blob/main/activations.py
# Activation Utils

import torch
from torch import nn
from torch import Tensor

def get_activation(activation_name: str, in_features: int = None) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leakyrelu": nn.LeakyReLU(),
        "snake": SnakeBeta(in_features)
    }
    return activations.get(activation_name.lower(), nn.Identity())


# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x
    
# Could be optimized for sure, I have a ton of CUDA OOM errors here...
def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)