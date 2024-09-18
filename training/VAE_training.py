# Code from Adhémar de Senneville
# AE Training Code

# Imports Torch
import torch
from torch import Tensor
import pytorch_lightning as pl

# Imports
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Imports Autre
import librosa
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import wandb

from models.autoencoders.encoder_1D import Autoencoder1d
from models.losses.factory import CraftLosses

class LitAutoEncoder(pl.LightningModule):
    def __init__(
            self,
            model_cfg: Dict,
            data_cfg: Dict,
            optimizer_cfg: Dict,
            loss_cfg: Dict
        ) -> None:
        super().__init__()
        self.save_hyperparameters() # for wandb
        
        # Model init
        self.model = Autoencoder1d(**model_cfg)
        
        # Training init
        self.generator_optimizer_cfg = optimizer_cfg['generator']
        self.discriminator_optimizer_cfg = optimizer_cfg['discriminator']
        self.sr = data_cfg['sr']
        self.loss = CraftLosses(**loss_cfg)

        # Placeholder for the first batch, first audio samples
        self.first_audio_sample = None
        self.log_audio_epoch = 1
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        info = self.model(x)
        info['x'] = x
        return info
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def training_step(self, batch, batch_idx):

        x = batch['x']
        
        # Log the original audio
        if self.first_audio_sample is None: 
            self.first_audio_sample = x[(0,),...]
        
        # Compute loss
        info = self.forward(x)
        log_dict = self.loss.backward(info)
        #loss = F.l1_loss(info['x'],info['x_hat']);loss.backward();log_dict = {'global_loss': loss}

        # Log average absolute gradient
        avg_gradients = torch.mean(torch.stack([torch.mean(torch.abs(p.grad)) for p in self.model.parameters() if p.grad is not None]))
        log_dict['generator_avg_gradient'] = avg_gradients

        # Log Balancer relative weights
        # TODO

        # Grad Update
        optimiser_ae, optimiser_discriminator = self.optimizers()

        optimiser_ae.step()
        optimiser_ae.zero_grad()

        if batch_idx % self.update_freq_discriminator == 0:
            log_dict['discriminator_loss'] = self.loss.backward_discriminator(info)

            # Log average absolute gradient
            avg_gradients = torch.mean(torch.stack([torch.mean(torch.abs(p.grad)) for p in self.loss.discriminator.parameters() if p.grad is not None]))
            log_dict['discriminator_avg_gradient'] = avg_gradients

            optimiser_discriminator.step()
            optimiser_discriminator.zero_grad()

        # Log
        self.log_dict(log_dict)
        return None
    
    def predict_step(self, batch, batch_idx):
        info = self.forward(batch['x'])
        return info

    def configure_optimizers(self):
        
        self.automatic_optimization = False
        self.update_freq_discriminator = self.discriminator_optimizer_cfg.pop('update_frequency')

        # Define two optimizers
        optimiser_ae = torch.optim.Adam(self.model.parameters(), **self.generator_optimizer_cfg)
        optimiser_discriminator = torch.optim.Adam(self.loss.discriminator.parameters(), **self.discriminator_optimizer_cfg)
        return [optimiser_ae, optimiser_discriminator]
    
    def on_train_epoch_end(self) -> None:
        
        if (self.first_audio_sample is not None) and (self.current_epoch % self.log_audio_epoch == 0) and hasattr(self.logger.experiment, 'log'):
            
            # Get the first audio sample
            original_audio = self.first_audio_sample[0][0].cpu().numpy()
            reconstructed_audio = self.forward(self.first_audio_sample)['x_hat'][0][0].cpu().detach().numpy()
            
            # Ensure the audio data is in the correct format
            if original_audio.dtype != 'float32':
                original_audio = original_audio.astype('float32')
            if reconstructed_audio.dtype != 'float32':
                reconstructed_audio = reconstructed_audio.astype('float32')
            
            # Log Audios
            if self.current_epoch==0:
                self.logger.experiment.log({
                    "original_audio": wandb.Audio(original_audio, sample_rate=self.sr, caption="Original Audio"),
                    "epoch": self.current_epoch
                })
            self.logger.experiment.log({
                "reconstructed_audio": wandb.Audio(reconstructed_audio, sample_rate=self.sr, caption="Reconstructed Audio"),
                "epoch": self.current_epoch
            })
            
            if False: # No need for spectrogram plotting for now
                ori_path = 'fig/original_spectrogram.png'
                rec_path = 'fig/reconstructed_spectrogram.png'
                # Compute spectrograms to decibel (dB) units
                original_spectrogram = librosa.stft(original_audio)
                reconstructed_spectrogram = librosa.stft(reconstructed_audio)
                original_spectrogram_db = librosa.amplitude_to_db(np.abs(original_spectrogram), ref=np.max)
                reconstructed_spectrogram_db = librosa.amplitude_to_db(np.abs(reconstructed_spectrogram), ref=np.max)

                # Log Spectrograms
                if self.current_epoch == 1:
                    plt.figure(figsize=(5, 5))
                    librosa.display.specshow(original_spectrogram_db, sr=self.sr, x_axis='time', y_axis='log')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Original Spectrogram')
                    plt.tight_layout()
                    plt.savefig(ori_path)
                    plt.close()

                    self.logger.experiment.log({
                        "original_spectrogram": wandb.Image(ori_path),
                        "epoch": self.current_epoch
                    })
                
                plt.figure(figsize=(5, 5))
                librosa.display.specshow(reconstructed_spectrogram_db, sr=self.sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Reconstructed Spectrogram')
                plt.tight_layout()
                plt.savefig(rec_path)
                plt.close()

                self.logger.experiment.log({
                    "reconstructed_spectrogram": wandb.Image(rec_path),
                    "epoch": self.current_epoch
                })