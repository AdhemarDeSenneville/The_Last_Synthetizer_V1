# Work in progress...

A first conditioning is a mask conditioning 

The Unet denoizer based on [DIFF-A-RIFF](https://arxiv.org/pdf/2406.08384)

[Encodec](https://arxiv.org/pdf/2210.13438)

The Auto-encoder compress 16 using [Snake](https://arxiv.org/pdf/2006.08195v2) activations 

Noize 
Some simple noise sheldurer exist like the cosin sine sheldurer used in Sable audio. I decided to go with the classic one used in the original paper

DDIM sheldurer known to be better (EDM) allow bigger step at generation sigma(t)=t and no scaling
ODE solver is just a simple 1rst order Euler scheam