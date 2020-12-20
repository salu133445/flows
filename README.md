# Flow-based Deep Generative Models

Authors: Jiarui Xu and Hao-Wen Dong

We investigate the flow-based deep generative models. We first compare different generative models, especially generative adversarial networks (GANs), variational autoencoders (VAEs) and flow-based generative models. We then survey different normalizing flow models, including non-linear independent components estimation (NICE), real-valued non-volume preserving (RealNVP) transformations, generative flow with invertible 1×1 convolutions (Glow), masked autoregressive flow (MAF) and inverse autoregressive flow (IAF). Finally, we conduct experiments on generating MNIST handwritten digits using NICE and RealNVP to examine the effectiveness of flow-based models.

## Notebooks

- RealNVP on toy dataset: `realnvp_toy.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salu133445/flows/blob/main/realnvp_toy.ipynb)
- RealNVP on MNIST: `realnvp_mnist.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salu133445/flows/blob/main/realnvp_mnist.ipynb)
