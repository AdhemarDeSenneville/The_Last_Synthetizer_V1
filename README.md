# Audio Latent Diffusion (Work in progress...)

[**Kaggle**](./code/Experiments_WaveShaper.ipynb)

## Work Overview

# Overview

This repo objectif is to be able to do neural synthesis of violons partitions using midi file or by composing the amplitude and notes of violons.
This is done using [Virtuoso Dataset](https://paperswithcode.com/dataset/virtuoso-strings) to trained an audio **latent diffusion model** conditionned on envelopes an pitches.

# Technical details

## Variational Auto-Encoder

The Auto-encoder compress 16 times the input by employing a architecture inspired by the on used by [Stable-Audio](https://github.com/Stability-AI/stable-audio-tools). More precisely mono input has a temporal resolution devided by 256 and channels multiplied by 16.
It uses ResNet-like 1D convolutions layers and [Snake](https://arxiv.org/pdf/2006.08195v2) activation. A bi-LSTM layer is added before the 2 last convolutions for further temporal receptiv field. Because the diffusion model in non streamable, having bi-LSTM is not problematic.
The two last convolutions compute the log variance and mean of the latent variable.

- **Temporal losses**: Using L1 and L2 recinstruction loss
- **Time-Frequency losses**: Using Multy-resolution time frequency losses
- **KL divergence**
- **Adversarial loss** 
Discriminator itself is trained on a hinge loss.

Instead of a simple summation of the losses, the loss balancer from [Encodec](https://arxiv.org/abs/2210.13438) was used. It has been changed so that it can monitor effective losses compare to other losses that are not usable in the balancer. This include losses in the latente space like KL divergence loss in the case of a VAE or Commitment and CodeBook losses in the case of a VQ-VAE.

## Diffusion model

The Unet denoizer based on the exact architecture of Multi-Scale Unet of [DIFF-A-RIFF](https://arxiv.org/pdf/2406.08384).
The diffusion framwork is the one from EDM known to reduce the number of time step at inference. The added noise follow $\sigma_{min} = 0.002, \sigma_{max} = 80$ and there is no scaling to the latent audio. Using a VAE I consider that $\sigma_{data} = 1$. For the sampling, $\rho = 7$, no stocasticity is introduced and ODE solver is 2nd order Heun.

### Conditionning

The time-step conditionning is the only conditionning used in the Unet. It uses classical sinusoidal embedding. I thought about using also class conditionning, however, classes in Virtuoso dataset are not locally distinguichable when lisenning to the data. It is then less relevante.

To other unconditionned conting are 'temporal' conditionning: envelope and pitch conditionning 
Similar to depth-conditionning used in images by [Stability AI](https://github.com/Stability-AI/stablediffusion) the pitch and envelope of the sound are concatenated to the latent noize befor enteling the network. Unconditional generation is performed according to a drop out probability. This allow for generation at inference witout needing a pitch of enveloppe input

As the envelope is temporaly copressed by 256 in the latent space, finally, I chose to compute it as the max amplitude over the window of 256 sample. The pitch detector use simply the max of a STFT.
