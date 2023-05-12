import torch
import torch.nn as nn

# basic building blocks common between many neural networks

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, use_norm=True):
        super(ConvBlock, self).__init__()
        self.use_norm = use_norm
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if self.use_norm: self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm: x = self.norm(x)
        return self.relu(x)


class LeakyConvBlock(ConvBlock):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, use_norm=True, slope=0.2):
        super(LeakyConvBlock, self).__init__(in_c, out_c, k, s, p, use_norm)
        self.relu = nn.LeakyReLU(slope, inplace=True)


class ConvTransBlock(nn.Module):
    def __init__(self, in_c, out_c, k=4, s=2, p=1, use_norm=True):
        super(ConvTransBlock, self).__init__()
        self.use_norm = use_norm
        self.conv_t = nn.ConvTranspose2d(in_c, out_c, kernel_size=k, 
                                         stride=s, padding=p, bias=False) 
        self.relu = nn.ReLU(inplace=True)
        if self.use_norm: self.norm = nn.BatchNorm2d(out_c)
    
    def forward(self, x):
        x = self.conv_t(x)
        if self.use_norm: x = self.norm(x)
        return self.relu(x)
