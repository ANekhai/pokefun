import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """PatchGAN inspired discriminator, deliberately unsophisticated"""
    def __init__(self):
        super(Discriminator, self).__init__()