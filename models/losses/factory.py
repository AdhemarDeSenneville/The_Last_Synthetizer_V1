# file name inspired by Stability AI


from balancer import Balancer
from losses import L1TemporalLoss, L2TemporalLoss, AuralossLoss, NoBalancer
from discriminator import Discriminator

cfg = {
    'balancer': True,
    'L1TemporalLoss':{'weight':1},
    'L2TemporalLoss':{'weight':1},
    'AuralossLoss':{'weight':1,
        "fft_sizes": [32, 128, 512, 2048],  # [32, 128, 512, 2048, 8192, 32768]
        "hop_sizes": [16, 64, 256, 1024],  # [16, 64, 256, 1024, 4096, 16384]
        "win_lengths": [32, 128, 512, 2048],  # [32, 128, 512, 2048, 8192, 32768]
        "w_sc": 0.0,
        "w_phs": 0.0,
        "w_lin_mag": 1.0,
        "w_log_mag": 1.0,
    },
    'Discriminator':{
        'weight':1,
        'type':'Spectrogram_VGGStyle',
    },
    'FeatureLoss':{
        'weight':1,
        
    },
    }

from torch import nn


class CraftLosses:

    def __init__(self, losses_config):

        is_balancer = losses_config.pop('balancer')

        self.balancer_config = {}
        self.losses = []

        for loss_name, loss_config in losses_config.items():
            weight = loss_config.pop('weight')
            loss = globals()[loss_name](*loss_config)

            self.losses.append(loss)
            self.balancer_config[loss.name] = weight

        if is_balancer:
            self.balancer = Balancer(self.balancer_config)  # Initialize balancer if specified
        else:
            self.balancer = NoBalancer(self.balancer_config)

    def forward(self, info):
        total_loss = 0.0

        all_losses = {}
        for loss in self.losses:
            all_losses[loss.name] = loss(info)

        total_loss = self.balancer(all_losses)

        return total_loss
