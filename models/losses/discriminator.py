# Code from Adhémar de Senneville
# Close to Encodec with 1 frequency resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

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
                nn.Conv2d(4, C, kernel_size=kernel_1, stride=1, padding=padding_1),
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

        freq_channels = fft_size//2 + 1 #- 2*1

        self.reduct_conv = nn.Sequential(
                nn.Conv2d(C, 1, kernel_size=3),
                nn.LeakyReLU()
            )
        self.final_conv = nn.Conv1d(501, 1, kernel_size=3)

    def forward(self, x):
        
        # Apply STFT, which converts (B, 1, T) -> (B, 2, F, T')
        x = self.stft(x)
        x = torch.stack([x[:,0,...].real, x[:,0,...].imag], dim=1)
        

        features = {}
        for l, block in enumerate(self.blocks):
            x = block(x)
            print(l, x.shape)
            features[l] = x

        x = self.reduct_conv(x).squeeze(1)
        print(x.shape)
        x = self.final_conv(x).squeeze(1)
        print(x.shape)

        return x
    
    def loss(self, x, x_hat):
        
        # Generator and Dicriminator loss
        score_fake = self(x_hat)
        score_real = self(x)

        generator_loss = -score_fake.mean()
        discriminator_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()

        featur_loss = 0.0
        
        with torch.no_grad():
            x = self.stft(x)
            x = torch.stack([x[:,0,...].real, x[:,0,...].imag, x[:,1,...].real, x[:,1,...].imag], dim=1)
        
        x_hat = self.stft(x_hat)
        x_hat = torch.stack([x_hat[:,0,...].real, x_hat[:,0,...].imag, x_hat[:,1,...].real, x_hat[:,1,...].imag], dim=1)
        
        for block in self.blocks:
            with torch.no_grad():
                x = block(x)
            x_hat = block(x_hat)

            featur_loss += F.l1_loss(x, x_hat)/torch.norm(x,1)
        
        return {
            'feature_loss': featur_loss,
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss + generator_loss, # TO SCALE GEN LOSS ACCORDINGLY IN THE BALANCER
        }


    
if __name__ == "__main__":
    # Test input: batch of audio signals with shape (B, 2, T)
    B, T = 4, 44000  # Batch size 4, 16000 samples (1 second at 16 kHz)
    test_input = torch.randn(B, 2, T)  # Random tensor simulating audio input
    test_hattt = torch.randn(B, 2, T)  # Random tensor simulating audio input

    model = Discriminator()  # Initialize the model

    # Perform inference
    output = model(test_input)

    print(model.loss(test_input,test_hattt))


    # Print the output shape
    print("Output shape:", output.shape)
