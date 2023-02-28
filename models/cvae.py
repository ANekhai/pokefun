import torch
from torch import nn

class CVAE(nn.Module):

    # General outline:
    # Image -> encoder -> mu, sigma -> 
    def __init__(self):
        super(CVAE, self).__init__()

        self.encoder = None
        self.decoder = None

    
    def forward(self, image):
        # encode image
        # reparameterization trick
        # decode image
        pass
    
    

