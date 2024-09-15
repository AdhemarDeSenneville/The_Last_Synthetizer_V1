# Inspired by stable_audio_tools
# added AuralossLoss and NoBalancer, changed atributs

import typing as tp

import torch
from torch import nn
from .auraloss import MultiResolutionSTFTLoss


TensorDict = tp.Dict[str, torch.Tensor]

class LossModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()

        self.name = name

    def forward(self, info, *args, **kwargs):
        raise NotImplementedError


class L1TemporalLoss(LossModule):
    def __init__(self, key_output: str, key_target: str):
        super().__init__(name='l1_loss')

        self.key_output = key_output
        self.key_target = key_target
        self.loss = nn.L1Loss(reduction='mean')
    
    def forward(self, info):
        loss = self.loss(info[self.key_output], info[self.key_target]) 
        return loss
    

class L2TemporalLoss(LossModule):
    def __init__(self, key_output: str, key_target: str):
        super().__init__(name='l2_loss')

        self.key_output = key_output
        self.key_target = key_target
        self.loss = nn.MSELoss(reduction='mean')
    
    def forward(self, info):
        loss = self.loss(info[self.key_output], info[self.key_target]) 
        return loss


class AuralossLoss(LossModule):
    def __init__(self, key_output: str, key_target: str, **config):
        super().__init__(name = 'aura_loss')

        self.key_output = key_output
        self.key_target = key_target
        self.loss = MultiResolutionSTFTLoss(**config)

    def forward(self, info):
        loss = self.loss(info[self.key_output], info[self.key_target])
        return loss


class KLDivergenceLoss(LossModule):
    def __init__(self, key_mean: str, key_log_std: str):
        super().__init__(name='kl_divergence_loss')

        self.key_mean = key_mean
        self.key_log_std = key_log_std

    def forward(self, info):
        # Mean and log_std are the outputs of a Gaussian distribution, used for the KL divergence calculation
        mean = info[self.key_mean]
        log_std = info[self.key_log_std]
        
        # KL divergence between N(mean, std) and N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std), dim=-1)
        kl_loss = kl_loss.mean()  # Mean over the batch

        return kl_loss


