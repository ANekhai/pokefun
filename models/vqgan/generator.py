import torch
import torch.nn as nn
import sys
# sys.path.append("..")
from common.conv_blocks import ConvBlock, LeakyConvBlock, ConvTransBlock


class Encoder(nn.Module):
    """Attempt at a simple encoder for VQGAN model"""
    def __init__(self, in_c=3, out_c=8, num_filters=64):
        super(Encoder, self).__init__()
        nf = num_filters
        self.channels = [nf*2**i for i in range(5)]

        self.proj_in = LeakyConvBlock(in_c, nf, 1, 1, 0, use_norm=False)
        self.proj_out = LeakyConvBlock(nf*16, out_c, 1, 1, 0, use_norm=False)
        self.model = nn.Sequential(
            self.proj_in,
            # (b,nf,96,96)
            LeakyConvBlock(nf, nf*2, 4, 2, 1),
            # (b,nf*2,48,48)
            LeakyConvBlock(nf*2, nf*4, 4, 2, 1),
            # (b,nf*4,24,24)
            LeakyConvBlock(nf*4, nf*8, 4, 2, 1),
            # (b,nf*8,12,12)
            LeakyConvBlock(nf*8, nf*16, 5, 1, 0 ),
            # (b,nf*16,8,8)
            self.proj_out
        )   

    def forward(self, x):
        return self.model(x) # (b,nz_v,nz,nz) = (b,8,8,8) right now

# One idea to try: using learned color module from colorgan! 
class Decoder(nn.Module):
    """Simple decoder: x_hat = D(z_q)"""
    def __init__(self, in_c=8, out_c=3, num_filters=64):
        super(Decoder, self).__init__()

        nf = num_filters
        self.channels = [nf*2**i for i in range(4, -1, -1)]

        self.proj_in = ConvTransBlock(in_c, nf*16  , 1, 1, 0, use_norm=False)
        self.proj_out = nn.Sequential(
            nn.Conv2d(nf, out_c, 1, 1, 0, bias=False),
            nn.Tanh(),
        )
        
        self.model = nn.Sequential(
            self.proj_in,
            # (b,nf*16,8,8)
            ConvTransBlock(nf*16, nf*8, 5, 1, 0),
            # (b,nf*4,12,12)
            ConvTransBlock(nf*8, nf*4),
            # (b,nf*4,24,24)
            ConvTransBlock(nf*4, nf*2),
            # (b,nf*2,48,48)
            ConvTransBlock(nf*2, nf),
            # (b,nf,96,96)
            self.proj_out,
        )
    
    def forward(self, z):
        return self.model(z)
        