# Code from Adhémar de Senneville
# Simplified LSTM Module

import torch.nn as nn

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