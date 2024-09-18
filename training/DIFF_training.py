# Code from Adhémar de Senneville
# Diffusion Training Code
# Sample function inspired by https://github.com/NVlabs/edm/blob/main/generate.py

import torch
import pytorch_lightning as pl

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from models.autoencoders.WaveUNet import UNet1d
from training.VAE_training import LitAutoEncoder

example_config = {
    'num_steps': 18,
    'rho': 7,
    'sigma_min': 0.002,
    'sigma_max': 80,
    'dropout_cond_envelope': 0.2,
    'dropout_cond_pitch': 0.2,
}

class LitDiffusion(pl.LightningModule):
    
    def __init__(
            self,
            model_cfg: Dict,
            optimizer_cfg: Dict,
            diffusion_cfg: Dict,
            auto_encoder_ckpt: str,
        ) -> None:
        super().__init__()

        # Networks
        self.model = UNet1d(**model_cfg)
        self.auto_encoder = LitAutoEncoder.load_from_checkpoint(auto_encoder_ckpt)
        self.optimizer_cfg = optimizer_cfg

        # Sampling
        self.num_steps = diffusion_cfg['num_steps']
        self.rho = diffusion_cfg['rho']
        self.latent_channel = model_cfg['out_channels']

        # Training
        self.sigma_min = diffusion_cfg['sigma_min']
        self.sigma_max = diffusion_cfg['sigma_max']
        self.sigma_mean = (math.log(self.sigma_min) + math.log(self.sigma_max)) / 2
        self.sigma_std = (math.log(self.sigma_max) - math.log(self.sigma_min)) / math.sqrt(12)
        self.dropout_cond_envelope = diffusion_cfg['dropout_cond_envelope']
        self.dropout_cond_pitch = diffusion_cfg['dropout_cond_pitch']

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), **self.optimizer_cfg)

    def training_step(self, batch, batch_idx):

        x = batch['x']
        z_envelope = batch['x_envelope']
        z_pitch = batch['x_pitch']

        # conditional dropout
        if torch.rand(1).item() < self.dropout_cond_envelope:
            z_envelope = torch.zeros_like(z_envelope)
        if torch.rand(1).item() < self.dropout_cond_pitch:
            z_pitch = torch.zeros_like(z_pitch)
        
        # Encode (could be avoid using stored latent dataset)
        with torch.no_grad():
            z = self.auto_encoder.encoder(x)        
        
        # EDM noise
        rnd_normal = torch.randn([z.shape[0], 1, 1], device=z.device)
        sigma = (rnd_normal * self.sigma_std + self.sigma_mean).exp()
        weight = (sigma ** 2 + 1) / (sigma) ** 2
        n = torch.randn_like(z) * sigma

        # Compute denoising and loss
        z_cond_noise = torch.cat((z + n, z_pitch, z_envelope), dim=1) # Mask conditionning
        z_hat = self.model(z_cond_noise, sigma)
        loss = weight * ((z_hat - z) ** 2)

        # Log
        log_dict = {
            'train/loss': loss.detach(),
        }

        self.log_dict(log_dict)
        return loss
    
    def sample(
            self,
            time = 8,
            z_envelope = None,
            z_pitch = None,
            num_steps = None,
            rho=None,
        ):
        
        if num_steps is None:
            num_steps = self.num_steps
        if rho is None:
            rho = self.rho
        
        if z_envelope is None and z_pitch is None:
            z_sample_number = int(time*self.auto_encoder.sr/self.auto_encoder.model.eff_padding)
            z_envelope = [0]*z_sample_number
            z_pitch = [0]*z_sample_number
        elif z_envelope is None:
            z_sample_number = len(z_pitch)
            z_envelope = [0]*z_sample_number
        elif z_pitch is None:
            z_sample_number = len(z_envelope)
            z_pitch = [0]*z_sample_number


        z_init = torch.randn([1, self.latent_channel, z_sample_number], device=self.device)
        z_envelope = torch.tensor(z_envelope, device=self.device).unsqueeze(0).unsqueeze(0)
        z_pitch = torch.tensor(z_pitch, device=self.device).unsqueeze(0).unsqueeze(0)

        # Time step discretization
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop
        z_next = z_init.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            z_cur = z_next

            # Euler step
            z_cur_cond = torch.cat((z_cur, z_pitch, z_envelope), dim=1)
            denoised = self.model(z_cur_cond, t_cur).to(torch.float64)
            d_cur = (z_cur - denoised) / t_cur
            z_next = z_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction
            if i < num_steps - 1:
                z_next_cond = torch.cat((z_next, z_pitch, z_envelope), dim=1)
                denoised = self.model(z_next_cond, t_next).to(torch.float64)
                d_prime = (z_next - denoised) / t_next
                z_next = z_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        with torch.no_grad():
            x_hat = self.auto_encoder.decode(z_next)['x_hat']

        return x_hat[0]
