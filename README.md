# Audio Latent Diffusion (Work in progress...)

[**Kaggle**](https://www.kaggle.com/code/adhmardesenneville/last-synth-ae)

# Overview

This repo's final objective is to be able to do neural synthesis using MIDI files.  
For now, this is done using an **Audio Latent Diffusion Model** conditioned on envelopes and pitches.

# Technical details

## Dataset

[Virtuoso Dataset](https://paperswithcode.com/dataset/virtuoso-strings) is used, which consists of high-quality violin recordings. For training, the audio is mono and downsampled to 22050Hz to reduce computation cost while keeping good audio quality. The average duration of audio in the dataset is 55s, but random 8s crops are used for the training.

## Variational Auto-Encoder

The Auto-encoder compresses the input 16 times by employing an architecture inspired by the one used by [Stable-Audio](https://github.com/Stability-AI/stable-audio-tools). More precisely, the mono input has its temporal resolution divided by 256 and its channels multiplied by 16.  
It uses ResNet-like 1D convolution layers and [Snake](https://arxiv.org/pdf/2006.08195v2) activations. A bi-LSTM layer is added before the last two convolutions for further temporal receptive field expansion. Since the diffusion model is non-streamable, having a bi-LSTM is not problematic.  
The last two convolutions compute the log variance and mean of the latent variable.

### Losses

- **Temporal losses**: Using L1 and L2 reconstruction loss
- **Time-Frequency losses**: Using Multi-resolution time frequency loss
- **KL Divergence**: Regularization term in the latent space
- **Adversarial loss**: The Discriminator is a VGG style network that takes as input the STFT of the audio. The Discriminator itself is trained on a hinge loss
- **Perceptual loss**: Based on feature maps from the Discriminator

Instead of a simple summation of the losses, the loss balancer from [Encodec](https://arxiv.org/abs/2210.13438) is used. It has been changed so that it can monitor effective losses compared to other losses that are not usable in the balancer. This includes losses in the latent space like KL divergence loss in the case of a VAE or Commitment and CodeBook losses in the case of a VQ-VAE.

## Diffusion model

The U-Net denoiser is based on the architecture of the Multi-Scale U-Net from [DIFF-A-RIFF](https://arxiv.org/pdf/2406.08384). The diffusion framework is the one from [EDM](https://arxiv.org/pdf/2206.00364), known to reduce the number of timesteps at inference. The added noise follows $\sigma_{min} = 0.002, \sigma_{max} = 80$, while no scaling is applied to the latent audio. Using a VAE, I consider that $\sigma_{data} = 1$. For the sampling, $\rho = 7$, no stochasticity is introduced, and the ODE solver is 2nd-order Heun.

### Conditionning

The timestep conditioning is the only conditioning used in the U-Net, it uses classical sinusoidal embedding. I considered using class conditioning; however, classes in the Virtuoso dataset are not locally distinguishable when listening to the data, making it less relevant.

The two other conditioning are 'temporal' conditioning on the envelope and the pitch. Similarly to depth conditioning used on images by [Stability AI](https://github.com/Stability-AI/stablediffusion), the pitch and envelope of the sound are concatenated to the latent noise before entering the network. Unconditional generation is performed according to a dropout probability, allowing for generation at inference without needing pitch or envelope input.

As the envelope is temporally compressed by 256 in the latent space, I chose to compute it as the max amplitude over the window of 256 samples. The pitch detector simply uses the max of an STFT.
