# Code from Adhémar de Senneville
# file name inspired by Stability AI
# A bit complicated, but modularity will be useful for my future projects

from .balancer import Balancer, NoBalancer
from .losses import L1TemporalLoss, L2TemporalLoss, AuralossLoss, KLDivergenceLoss
from .discriminator import Discriminator

import torch
from torch import Tensor
import typing as tp

example_cfg = {
    'L1TemporalLoss':{
        'weight':1,
        'balancer': True,
        'key_output': 'x_hat',
        'key_target': 'x',
        },
    'L2TemporalLoss':{
        'weight':1,
        'balancer': True,
        'key_output': 'x_hat',
        'key_target': 'x',
        },
    'AuralossLoss':{
        'weight':1,
        'balancer': True,
        'key_output': 'x_hat',
        'key_target': 'x',
        "fft_sizes": [32, 128, 512, 2048],  # [32, 128, 512, 2048, 8192, 32768]
        "hop_sizes": [16, 64, 256, 1024],  # [16, 64, 256, 1024, 4096, 16384]
        "win_lengths": [32, 128, 512, 2048],  # [32, 128, 512, 2048, 8192, 32768]
        "w_sc": 0.0,
        "w_phs": 0.0,
        "w_lin_mag": 1.0,
        "w_log_mag": 1.0,
    },
    'KLDivergenceLoss':{
        'weight':1,
        'balancer': False,
        'key_mean': 'z_mean',
        'key_log_std': 'z_log_std',
        },
    'Discriminator':{
        'weight':1,
        'balancer': True,
        #'type':'Spectrogram_VGGStyle',
    },
    'FeatureLoss':{
        'weight':1,
        'balancer': True,
    },
    }

from torch import nn


class CraftLosses(nn.Module):

    def __init__(self, **losses_config) -> None:
        super().__init__()

        self.balancer_config = {}
        self.nobalancer_config = {}
        self.losses = []

        # Init discriminator network
        discriminator_config = losses_config.pop('Discriminator')
        weight = discriminator_config.pop('weight')
        is_balancer = discriminator_config.pop('balancer')
        self.discriminator = Discriminator(**discriminator_config)

        # Generator loss
        if is_balancer: self.balancer_config['generator_loss'] = weight
        else: self.nobalancer_config['generator_loss'] = weight

        # Feature loss
        feature_loss_config = losses_config.pop('FeatureLoss')
        if is_balancer: self.balancer_config['feature_loss'] = feature_loss_config['weight']
        else: self.nobalancer_config['feature_loss'] = feature_loss_config['weight']
        
        # Other losses
        for loss_name, loss_config in losses_config.items():

            weight = loss_config.pop('weight')
            is_balancer = loss_config.pop('balancer')

            loss = globals()[loss_name](**loss_config)
            self.losses.append(loss)

            if is_balancer:
                self.balancer_config[loss.name] = weight
            else:
                self.nobalancer_config[loss.name] = weight

        self.balancer = Balancer(self.balancer_config)
        self.nobalancer = NoBalancer(self.nobalancer_config)

    def backward(self, info: tp.Dict[str, Tensor], logging: bool = True) -> tp.Optional[tp.Dict[str, Tensor]]:

        all_losses = self.discriminator.generator_loss(info['x'], info['x_hat'])

        for loss in self.losses:
            all_losses[loss.name] = loss(info)
        
        if len(self.nobalancer_config.keys()) != 0:
            effective_losses_nobalancer = self.nobalancer.backward({key: all_losses[key] for key in self.nobalancer_config}, retain_graph=True)
        else:
            effective_losses_nobalancer = {}
        
        if len(self.balancer_config.keys()) != 0:
            effective_losses_balancer = self.balancer.backward({key: all_losses[key] for key in self.balancer_config}, info['x_hat'])
        else:
            effective_losses_balancer = {}

        if logging:
            effective_losses = {**effective_losses_nobalancer, **effective_losses_balancer}
            effective_losses['global_loss'] = sum(effective_losses.values())
            return effective_losses
    
    def backward_discriminator(self, info: tp.Dict[str, Tensor], logging: bool = True) -> tp.Optional[Tensor]:

        discriminator_loss = self.discriminator.discriminator_loss(info['x'], info['x_hat'])
        discriminator_loss.backward()

        if logging:
            return discriminator_loss
