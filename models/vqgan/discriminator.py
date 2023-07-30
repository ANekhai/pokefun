import torch
import torch.nn as nn
import sys
# sys.path.append("..")
from common.conv_blocks import LeakyConvBlock

class Discriminator(nn.Module):
    """PatchGAN inspired discriminator, deliberately unsophisticated"""
    def __init__(self, in_c=3, out_c=1, num_filters=64):
        super(Discriminator, self).__init__()
        nf = num_filters

        self.proj_in  = LeakyConvBlock(in_c, nf, 4, 2, 1, use_norm=False)
        self.proj_out = nn.Sequential(
            nn.Conv2d(nf*8, out_c, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            # (b,nf,48,48)
            LeakyConvBlock(nf, nf*2, 4, 2, 1),
            # (b,nf*2,24,24)
            LeakyConvBlock(nf*2, nf*4, 4, 2, 1),
            # (b,nf*4,12,12)
            LeakyConvBlock(nf*4, nf*8, 4, 2, 1)
            # out: (b,nf*8,6,6)
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.conv(x)
        logits = self.proj_out(x)
        return logits
