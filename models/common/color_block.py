import torch
import torch.nn as nn


class ColorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, z_dim):
        super(ColorBlock, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels


        self.z_to_color = nn.Sequential(
            # (b, z_dim)
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, 128)
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #(b, 64)
            nn.Linear(64, self.in_c*self.out_c),
            nn.Tanh(),
            # (b, in_c * out_c)
        )

        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, im, z):
        
        # get colorspace
        colorspace = self.z_to_color(z).view(-1, 1, self.in_c, self.out_c)

        # (b, in_c, x, y) -> (b, y, x, in_c)
        im.transpose_(1, 3) 
        # (b, y, x, in_c) @ (b, 1, in_c, out_c) -> (b, y, x, out_C)
        logits = im @ colorspace
        logits.transpose_(1, 3)
        return logits
    