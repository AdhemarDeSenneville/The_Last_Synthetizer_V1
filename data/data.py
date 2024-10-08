
"""
Code from Adhémar de Senneville

AudioDataset class

Components:
- AudioDataset: Main class that manages the loading, processing, and data retrieval.
  - __init__: Initializes the dataset and its configurations.
  - _get_envelope: Extracts the envelope of the audio signal using the specified detection method.
  - _get_pitch: Extracts the pitch of the audio signal using librosa and interpolates the result.
  - __getitem__: Returns a dictionary containing waveform, envelope, and pitch for a given index.
  - plot_item: Visualizes the waveform, envelope, and pitch for a selected audio sample and plays the audio.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import torch
import torchaudio
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
import soundfile as sf
import random
from math import ceil
import IPython.display as ipd


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, **config):

        self.data_path = config['path']
        self.sr = config['sr']
        self.sample_duration = config['sample_duration']
        self.latent_compression = config['latent_compression']

        # Envelope Detector 
        self.envelope_detector_config = config['envelope_detector']
        self.envelope_detector_type = self.envelope_detector_config.pop('type')
        if self.envelope_detector_type == 'hilbert':
            b, a = butter(N=5, Wn=self.envelope_detector_config['low_pass'] / (0.5 * self.sr), btype='low')
            self.envelope_detector_config['a'] = a
            self.envelope_detector_config['b'] = b
        
        # Pitch Detector 
        self.pitch_detector_config = config['pitch_detector']

        self.wav_files = self._get_wav_files()
    
    def _get_wav_files(self):
        wav_files = []

        for root, dirs, files in os.walk(self.data_path):
            
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        return wav_files
    
    def _get_envelope(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the envelope of the audio signal using the specified envelope detection method.

        Parameters:
        x (np.ndarray): The input audio signal. Shape: (T,)

        Returns:
        np.ndarray: The computed envelope of the audio signal. Shape: (ceil(T/latent_compression),)
        """

        pad_length = (self.latent_compression - (len(x) % self.latent_compression)) % self.latent_compression
        if pad_length > 0:
            x = np.pad(x, (0, pad_length), mode='constant')
        
        # Envelope
        if self.envelope_detector_type == 'hilbert':
            x_envelope = filtfilt(self.envelope_detector_config['b'], self.envelope_detector_config['a'], x)
            x_envelope = hilbert(x_envelope)
            x_envelope = np.abs(x_envelope)
            x_envelope = x_envelope.reshape(-1, self.latent_compression)
            x_envelope = np.mean(x_envelope, axis=1)

        elif self.envelope_detector_type == 'RMS':
            x_envelope = x.reshape(-1, self.latent_compression)
            x_envelope = np.std(x_envelope, axis=1)*np.sqrt(2)

        elif self.envelope_detector_type == 'Max':
            x_envelope = x.reshape(-1, self.latent_compression)
            x_envelope = np.max(x_envelope, axis=1)
        
        
        return x_envelope


    
    def _get_pitch(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the pitch of the audio signal by using the librosa piptrack method and interpolating the result.

        Parameters:
        x (np.ndarray): The input audio signal. Shape: (T,)

        Returns:
        np.ndarray: The computed pitch of the audio signal. Shape: (ceil(T/latent_compression),)
        """
    
        # Extract pitch from  magnitude
        #pitches, _ = librosa.piptrack(y=x, sr=self.sr, **self.pitch_detector_config)
        #x_pitch = np.max(pitches, axis=0)
        
        # Interpolate
        #original_len = len(x_pitch)
        #target_len = ceil(len(x) / self.latent_compression)
        #interpolator = interp1d(np.arange(original_len), x_pitch, kind='linear', fill_value="extrapolate")
        #x_pitch = interpolator(np.linspace(0, original_len - 1, target_len))

        pitches, magnitudes = librosa.core.piptrack(
            y=x, 
            sr=self.sr, 
            hop_length=self.latent_compression, 
            center=True, 
            **self.pitch_detector_config,
        )
        max_indexes = np.argmax(magnitudes, axis=0)
        x_pitch = pitches[max_indexes, range(magnitudes.shape[1])]

        return x_pitch

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx: int):
        wav_path = self.wav_files[idx]
        # waveform, sample_rate = torchaudio.load(wav_path) # Strange torch - numpy incompatibility
        waveform, sample_rate = sf.read(wav_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Resample to self.sr if sample rates differ
        if sample_rate != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(waveform)

        # Random part that has self.sample_duration
        num_samples = int(self.sample_duration * self.sr)
        start_idx = random.randint(0, waveform.size(0) - num_samples)
        waveform = waveform[start_idx:start_idx + num_samples]
        waveform_np = np.array(waveform)
        
        # Get the envelope and pitch
        x_envelope = self._get_envelope(waveform_np)
        x_pitch = self._get_pitch(waveform_np)

        return {
            'x': waveform.unsqueeze(0),
            'x_envelope': torch.tensor(x_envelope).unsqueeze(0),
            'x_pitch': torch.tensor(x_pitch).unsqueeze(0)
        }
    

    def plot_item(self, i: int):
        """
        Plots the waveform, envelope, and pitch of the audio sample at index i.

        Parameters:
        i (int): Index of the audio sample to plot.
        """
        
        # Extract waveform, envelope, and pitch
        item = self[i]
        waveform = item['x'].numpy()[0]
        envelope = item['x_envelope'].numpy()[0]
        pitch = item['x_pitch'].numpy()[0]
        t = np.linspace(0, len(waveform) / self.sr, len(waveform))
        
        # Plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(t, waveform, label='Waveform', alpha=0.7)
        axs[0].plot(t[::self.latent_compression], envelope, label='Envelope', color='orange', alpha=0.7)
        axs[0].set_title('Waveform and Envelope')
        axs[0].legend(loc='upper right')
        
        axs[1].plot(t[::self.latent_compression], pitch)
        axs[1].set_title('Pitch')
        
        plt.tight_layout()
        plt.show()
        ipd.Audio(waveform, rate=self.sr)