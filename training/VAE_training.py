
# Imports
import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import time
import sys
import gc
import yaml
import pickle
from math import floor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Imports Torch
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

# Imports Autre
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import IPython.display as ipd
from matplotlib import pyplot as plt
import wandb
from pytorch_lightning.loggers import WandbLogger
#import seaborn as sns

from models.autoencoders.encoder_1D import Autoencoder1d
from models.losses.factory import CraftLosses

class LitAutoEncoder(pl.LightningModule):
    def __init__(
            self,
            model_cfg,
            data_cfg,
            optimizer_cfg,
            loss_cfg):
        super(LitAutoEncoder, self).__init__()
        self.save_hyperparameters() # for wandb
        
        # Model init
        self.model = Autoencoder1d(**model_cfg)
        
        # Training init
        self.optimizer_cfg = optimizer_cfg
        self.sr = data_cfg['sr']
        self.loss = CraftLosses(**loss_cfg)

        # Placeholder for the first batch, first audio samples
        self.first_audio_sample = None
        
    def forward(self, x):
        info = self.model(x)
        info['x'] = x
        return info
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def training_step(self, batch, batch_idx):

        x = batch['x']
        
        # Log the original audio
        if self.first_audio_sample is None: 
            self.first_audio_sample = x[(0,),...]
        
        # Compute loss
        info = self.forward(x)

        # Grad Update
        optimiser_ae, optimiser_discriminator = self.optimizers()
        
        log_dict = self.loss.backward(info)
        optimiser_ae.step()
        optimiser_ae.zero_grad()

        if batch_idx % self.update_freq_discriminator == 0:
            log_dict['discriminator_loss'] = self.loss.backward_discriminator(info)
            optimiser_discriminator.step()
            optimiser_discriminator.zero_grad()

        # Log
        self.log_dict(log_dict) #, on_step=True, on_epoch=True
        return log_dict
    
    def predict_step(self, batch, batch_idx):
        info = self.forward(batch['x'])
        return info

    def configure_optimizers(self):
        self.automatic_optimization = False  # Use manual optimization
        self.update_freq_discriminator = 2
        # Define two optimizers
        optimiser_ae = torch.optim.Adam(self.model.parameters(), **self.optimizer_cfg)
        optimiser_discriminator = torch.optim.Adam(self.loss.discriminator.parameters(), **self.optimizer_cfg)
        return [optimiser_ae, optimiser_discriminator]
    
    def on_train_epoch_end(self):        
        #print('here')
        
        if (self.first_audio_sample is not None) and (self.current_epoch % 10 == 1) and hasattr(self.logger.experiment, 'log'):
            
            # Get the first audio sample
            #print(self.first_audio_sample.device)
            original_audio = self.first_audio_sample[0].cpu().numpy()
            reconstructed_audio = self.forward(self.first_audio_sample)['x_hat'][0][0].cpu().detach().numpy()
            
            # Ensure the audio data is in the correct range and format
            if original_audio.dtype != 'float32':
                original_audio = original_audio.astype('float32')
            if reconstructed_audio.dtype != 'float32':
                reconstructed_audio = reconstructed_audio.astype('float32')

            # Normalize audio to be in the range -1.0 to 1.0
            # original_audio /= np.max(np.abs(original_audio), axis=-1, keepdims=True)
            # reconstructed_audio /= np.max(np.abs(reconstructed_audio), axis=-1, keepdims=True)
            # print(original_audio.shape)
            # print(reconstructed_audio.shape)
            
            # Log Audios
            if self.current_epoch==1:
                self.logger.experiment.log({
                    "original_audio": wandb.Audio(original_audio[0], sample_rate=self.sr, caption="Original Audio"),
                    "epoch": self.current_epoch
                })
            self.logger.experiment.log({
                "reconstructed_audio": wandb.Audio(reconstructed_audio[0], sample_rate=self.sr, caption="Reconstructed Audio"),
                "epoch": self.current_epoch
            })
            
            ori_path = 'fig/original_spectrogram.png'
            rec_path = 'fig/reconstructed_spectrogram.png'
            # Compute spectrograms to decibel (dB) units
            original_spectrogram = librosa.stft(original_audio[0])
            reconstructed_spectrogram = librosa.stft(reconstructed_audio[0])
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