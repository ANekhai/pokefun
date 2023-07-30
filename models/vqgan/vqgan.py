import torch
import torch.nn as nn
from .generator import Encoder, Decoder
from .discriminator import Discriminator
from .quantization import Codebook
import sys
# sys.path.append("..")
from common.conv_blocks import ConvBlock

class VQGAN(nn.Module):
    """VQGAN implementation using simplified generator, discriminator, and perceptual loss
       Aims to learn a latent representation nz"""
    def __init__(self, codebook_size, latent_dim, beta):
        """Attempt at simplified implementation of VQGAN"""
        super(VQGAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.codebook = Codebook(codebook_size, latent_dim, beta)

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
        z = self.proj_in(z)
        z_q, ind, q_loss = self.codebook(z)
        z_q = self.proj_out(z_q)
        x_hat = self.decoder(z_q)
    
        return x_hat, ind, q_loss
    

    def encode(self, im):
        """f: x -> z"""
        pass


    def decode(self, z):
        """f: z -> x_hat"""
        pass
    

    @staticmethod
    def get_discrim_discount(factor, steps, threshold, value=0.):
        return factor if steps >= threshold else value