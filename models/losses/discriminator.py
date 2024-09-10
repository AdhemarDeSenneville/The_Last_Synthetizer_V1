import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

class AudioDiscriminator(nn.Module):
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024):
        super(AudioDiscriminator, self).__init__()

        # STFT parameters
        self.stft = transforms.Spectrogram(
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            power=None  # return complex values (real + imag)
        )

        C = 32
        kernel_1 = (3,9)
        stride_1 = (1,2)

        # VGG-style discriminator layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, C, kernel_size=kernel_1, stride=1),  # First layer: input 2 channels (real/imag from STFT)
            nn.BatchNorm2d(C),
            nn.LeakyReLU(),

            nn.Conv2d(C,C, kernel_size=kernel_1, stride=stride_1, dilation=(1,1)),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(),

            nn.Conv2d(C, C, kernel_size=kernel_1, stride=stride_1, dilation=(2,1)),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(),

            nn.Conv2d(C, C, kernel_size=kernel_1, stride=stride_1, dilation=(4,1)),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(),

            nn.Conv2d(C, C, kernel_size=3),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(),

            nn.Conv2d(C, 1, kernel_size=3),
        )

    def forward(self, x):
        # x shape: (B, 2, T)
        # Apply STFT, which converts (B, 2, T) -> (B, 2, F, T')
        x = self.stft(x)
        x = torch.stack([x[:,0,...].real, x[:,0,...].imag, x[:,1,...].real, x[:,1,...].imag], dim=1)

        # Pass through VGG-style convolutional layers
        x = self.conv_layers(x)

        return x


class FeatureLoss(nn.Module):
    def __init__(self, feature_extractor, target_layer_indices=None):
        super(FeatureLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.target_layer_indices = target_layer_indices or [2, 4, 6]  # Define which layers to extract features from

    def forward(self, x, target):
        # Extract features from both input and target using the feature extractor
        x_features = self._extract_features(x)
        target_features = self._extract_features(target)

        # Compute L1 loss between corresponding feature maps
        loss = 0.0
        for x_f, t_f in zip(x_features, target_features):
            loss += F.l1_loss(x_f, t_f)

        return loss

    def _extract_features(self, x):
        features = []
        for i, layer in enumerate(self.feature_extractor.conv_layers):
            x = layer(x)
            if i in self.target_layer_indices:
                features.append(x)
        return features


if __name__ == "__main__":
    # Test input: batch of audio signals with shape (B, 2, T)
    B, T = 4, 16000  # Batch size 4, 16000 samples (1 second at 16 kHz)
    test_input = torch.randn(B, 2, T)  # Random tensor simulating audio input

    model = AudioDiscriminator()  # Initialize the model

    # Perform inference
    output = model(test_input)

    # Print the output shape
    print("Output shape:", output.shape)
