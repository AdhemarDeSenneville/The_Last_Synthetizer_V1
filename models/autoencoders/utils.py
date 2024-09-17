# Code from Adhémar de Senneville
# For similar size after bottle neck architecture

import torch.nn.functional as F
from torch import Tensor

def pre_process(x: Tensor, eff_padding: int):
    """
    Zero-pad the temporal dimension (dimension 2) to make its length a multiple of eff_padding.
    """
    x_length = x.shape[-1]
    pad_length = (eff_padding - (x_length % eff_padding)) % eff_padding

    if pad_length > 0:
        x = F.pad(x, (0, pad_length))

    return x, x_length

def post_process(x: Tensor, x_length: int):
    """
    Remove any excess padding in the temporal dimension to restore the original length.
    """
    return x[..., :x_length]