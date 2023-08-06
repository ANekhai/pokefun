import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Major change between tutorial happens here as I need to end up with a 3x96x96 image
            nn.ConvTranspose2d( nz, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # Original model used an inplace ReLU, ganhacks suggests using leakyReLU in both G and D
            nn.LeakyReLU(0.2, inplace=True), 
            # state size: (ngf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*2) x 24 x 24
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf) x 48 x 48
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Softmax(dim=1)
            # state size: (nc) x 96 x 96
        )