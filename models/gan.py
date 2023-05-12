import torch
import torch.nn as nn
import torch.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, )


class NoisyConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 kernel_size=4, stride=2, padding=1, device="cuda"):
        super(NoisyConvBlock, self).__init__()
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv_net(x) + torch.randn_like(x, device=self.device)
        return x

# Idea: Add residual layers in generator


