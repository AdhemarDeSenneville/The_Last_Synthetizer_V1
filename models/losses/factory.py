# Code from Adhémar de Senneville
# file name inspired by Stability AI
# A bit complicated, but modularity will be useful for my future projects

from .balancer import Balancer, NoBalancer
from .losses import L1TemporalLoss, L2TemporalLoss, AuralossLoss, KLDivergenceLoss
from .discriminator import Discriminator


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

    def __init__(self, **losses_config):

        self.balancer_config = {}
        self.nobalancer_config = {}
        self.losses = []

        # Discriminator loss
        discriminator_config = losses_config.pop('Discriminator')
        weight = discriminator_config.pop('weight')
        is_balancer = discriminator_config.pop('balancer')
        self.discriminator = Discriminator(**discriminator_config)
        if is_balancer: self.balancer_config['discriminator_loss'] = weight
        else: self.nobalancer_config['discriminator_loss'] = weight

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

    def backward(self, info, logging = True):

        all_losses = self.discriminator.loss(info['x'], info['x_hat'])

        for loss in self.losses:
            all_losses[loss.name] = loss(info)

        self.all_losses = all_losses
        
        self.nobalancer.backward({key: all_losses[key] for key in self.nobalancer_config}, retain_graph=True)
        self.balancer.backward({key: all_losses[key] for key in self.balancer_config}, info['x_hat'])

        if logging:
            return all_losses
