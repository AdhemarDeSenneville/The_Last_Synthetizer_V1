# Inspired by stable_audio_tools
# added AuralossLoss and NoBalancer, changed atributs

import typing as tp

import torch
from torch import nn
from auraloss import MultiResolutionSTFTLoss


TensorDict = tp.Dict[str, torch.Tensor]

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()

        self.name = name

    def forward(self, info, *args, **kwargs):
        raise NotImplementedError


class L1TemporalLoss(LossModule):
    def __init__(self, key_output: str, key_target: str, name: str = 'l1_loss'):
        super().__init__(name=name)

        self.key_output = key_output
        self.key_target = key_target
        self.loss = nn.L1Loss(reduction='mean')
    
    def forward(self, info):
        self.loss = self.loss(info[self.key_output], info[self.key_target]) 
        return self.loss
    

class L2TemporalLoss(LossModule):
    def __init__(self, key_output: str, key_target: str, name: str = 'L2_loss'):
        super().__init__(name=name)

        self.key_output = key_output
        self.key_target = key_target
        self.loss = nn.MSELoss(reduction='mean')
    
    def forward(self, info):
        self.loss = self.loss(info[self.key_output], info[self.key_target]) 
        return self.loss


class AuralossLoss(LossModule):
    def __init__(self, *config, key_output: str, key_target: str, name: str):
        super().__init__(name)

        self.key_output = key_output
        self.key_target = key_target
        self.loss = MultiResolutionSTFTLoss(*config)

    def forward(self, info):
        loss = self.loss(info[self.key_output], info[self.key_target])
        return loss


class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses


class KLDivergenceLoss(LossModule):
    def __init__(self, key_mean: str, key_log_std: str, name: str = 'kl_divergence_loss'):
        super().__init__(name=name)

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


class NoBalancer:
    def __init__(self, weights: tp.Dict[str, float]):
        self.weights = weights

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor):

        total_loss = 0.0

        for name, value in losses.items():
            total_loss += value * self.weights[name]
        
        return total_loss
