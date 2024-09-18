# Code from Adhémar de Senneville
# Close to Encodec one with 1 frequency resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
from torch import Tensor

class Discriminator(nn.Module):
    def __init__(self, fft_size=1024, hop_frac=1/4, win_length=1024):
        super().__init__()
        self.name = 'discriminator'

        # STFT parameters
        self.stft = transforms.Spectrogram(
            n_fft=fft_size,
            hop_length=int(fft_size*hop_frac),
            win_length=win_length,
            power=None  # return complex values (real + imag)
        )

        C = 32
        kernel_1 = (3,9)
        padding_1 = (1,4)
        stride_1 = (1,2)

        # VGG-style discriminator layers
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, C, kernel_size=kernel_1, stride=1, padding=padding_1),
                nn.BatchNorm2d(C),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=kernel_1, stride=stride_1, dilation=(1, 1), padding=padding_1),
                nn.BatchNorm2d(C),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=kernel_1, stride=stride_1, dilation=(2, 1), padding=padding_1),
                nn.BatchNorm2d(C),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=kernel_1, stride=stride_1, dilation=(4, 1), padding=padding_1),
                nn.BatchNorm2d(C),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3),
                nn.BatchNorm2d(C),
                nn.LeakyReLU()
            )
        ])


        self.reduct_conv = nn.Sequential(
                nn.Conv2d(C, 1, kernel_size=3),
                nn.LeakyReLU()
            )
        
        # freq_channels = fft_size//2 + 1 #- 2*1
        self.final_conv = nn.Conv1d(501, 1, kernel_size=3) # Not in Encodec

    def forward(self, x: Tensor) -> Tensor:

        self.stft = self.stft.to(x.device)
        
        # Apply STFT, which converts (B, 1, T) -> (B, 2, F, T')
        x = self.stft(x)
        x = torch.stack([x[:,0,...].real, x[:,0,...].imag], dim=1)
        
        for l, block in enumerate(self.blocks):
            # For debugging
            x = block(x)

        x = self.reduct_conv(x).squeeze(1)
        x = self.final_conv(x).squeeze(1)

        return x

    def discriminator_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:

        score_fake = self(x_hat.detach())
        score_real = self(x)

        discriminator_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()

        return discriminator_loss
    
    def generator_loss(self, x: Tensor, x_hat: Tensor):

        feature_loss = 0.0
        
        with torch.no_grad(): # Decreases computation cost
            x = self.stft(x)
            x = torch.stack([x[:,0,...].real, x[:,0,...].imag], dim=1)
        
        x_hat = self.stft(x_hat)
        x_hat = torch.stack([x_hat[:,0,...].real, x_hat[:,0,...].imag], dim=1)
        
        for block in self.blocks:
            with torch.no_grad():
                x = block(x)
            x_hat = block(x_hat)

            feature_loss += torch.norm(x - x_hat,1)/torch.norm(x,1)
        
        x_hat = self.reduct_conv(x_hat).squeeze(1)
        score_fake = self.final_conv(x_hat).squeeze(1)
        
        generator_loss = -score_fake.mean()
        
        return {
            'feature_loss': feature_loss,
            'generator_loss': generator_loss,
        }


    
if __name__ == "__main__":
    # Test input: batch of audio signals with shape (B, 1, T)
    B, T = 4, 44000 
    test_input = torch.randn(B, 1, T)
    test_hattt = torch.randn(B, 1, T)

    model = Discriminator()

    # Perform inference
    output = model(test_input)
    print(model.generator_loss(test_input,test_hattt))
    print(model.discriminator_loss(test_input,test_hattt))

    # Print the output shape
    print("Output shape:", output.shape)
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))