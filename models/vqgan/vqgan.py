import torch
import torch.nn as nn
from .generator import Encoder, Generator
from .discriminator import Discriminator
from .quantization import Codebook
import sys
sys.path.append("..")
from common.utils import ConvBlock

class VQGAN(nn.Module):
    """VQGAN implementation using simplified generator, discriminator, and perceptual loss
       Aims to learn a latent representation nz"""
    def __init__(self, codebook_size, latent_dim):
        """Attempt at simplified implementation of VQGAN"""
        super(VQGAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Generator()
        self.discriminator = Discriminator()
        self.codebook = Codebook(codebook_size, latent_dim)

        self.proj_in = nn.Sequential(
            nn.Dropout(),
            ConvBlock(latent_dim, latent_dim, 1, 1, 0, use_norm=False),
            )

        self.proj_out = nn.Sequential(
            nn.Dropout(),
            ConvBlock(latent_dim, latent_dim, 1, 1, 0, use_norm=False),
            )

    def forward(self, x):
        z = self.encoder(x)
        z_q, ind, q_loss = self.codebook(z)
        x_hat = self.generator(z_q)
    
        return x_hat, ind, q_loss