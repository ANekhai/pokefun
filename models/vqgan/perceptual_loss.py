import torch
import torch.nn as nn
from common.conv_blocks import ConvBlock

"""
The idea is to use a model trained on some alternate task to compare two images
EX: take vgg16 and extract intermediate layers and use those as loss
vgg16 is probably overkill for pokemon, so let's try with a dead simple type classifier instead
"""

class TypeClassifier(nn.Module):
    """Module to classify both types of a pokemon sprite"""
    def __init__(self, in_channels=3, num_features=64, depth=3, n_types=19):
        super(TypeClassifier, self).__init__()
        nc = in_channels
        nf = num_features
        self.n_types = n_types
        in_cs = [nf * 2 ** i for i in range(depth)]
        out_cs = [2*i for i in in_cs]

        # (nc,96,96) -> (nf,48,48)
        self.projection = ConvBlock(nc, nf, 4, 2, 1, use_norm=False)
        # (nf,48,48) -> (nf*2,24,24) -> (nf*4,12,12) -> (nf*8,6,6)
        self.convolutions = nn.ModuleList([ConvBlock(in_c, out_c, 4, 2, 1)
                                                    for in_c, out_c in zip(in_cs, out_cs)])
        # (nf*8,6,6) -> (n_types,1,1)
        self.head = nn.Sequential(
            nn.Conv2d(nf*8, self.n_types, 6, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.projection(x)
        x_embeds = [x]
        for conv in self.convolutions:
            x = conv(x)
            x_embeds.append(x)
        logits = self.head(x).view(-1, self.n_types)
        return logits, x_embeds

# Some ideas for how to improve above classifier
# 1. Have two separate classification heads w/ softmax activation to predict types
# 2. Try two models for primary and secondary types, concat embeddings for loss
# 3. Fancier blocks weren't useful for such a small dataset, overfitting started earlier w/ fancier models

class PerceptualLoss(nn.Module):
    """Module to compare a pokemon sprite to its recreation"""
    def __init__(self, pt_path="pretrained_models/ploss.pt"):
        super(PerceptualLoss, self).__init__()
        self.model = TypeClassifier()

        # load model
        checkpoint = torch.load(pt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
        for param in self.parameters(): param.requires_grad = False

    def forward(self, x_real, x_fake):
        _, e_real = self.model(x_real)
        _, e_fake = self.model(x_fake)

        losses = torch.zeros(len(e_real))
        # concatenate outputs
        for i, es in enumerate(zip(e_real, e_fake)):
            real, fake = es
            losses[i] = torch.mean((real-fake)**2)
            

        return torch.mean(losses)


# Again, we operate under the assumption VGG et al. are overkill for our puny pokemon dataset
# Let's try to implement LPIPS, which uses convolutional fuzzing and some fancy operations
# Might be overkill, we'll explore later
class LPIPS(nn.Module):
    """Follow LPIPS paper to implement, video uses this loss"""
    def __init__(self, pt_path="pretrained_models/ploss.pt", eval_mode=True):
        super(LPIPS, self).__init__()
        self.model = TypeClassifier()

        # load model
        checkpoint = torch.load(pt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.channels = [64 * 2 ** i for i in range(4)]

        # noise applying models, left untrained
        self.nets = nn.ModuleList([LossLinLayer(c) for c in self.channels])

        if eval_mode: self.model.eval()
        
    def forward(self, real_x, fake_x):
        _, e_real = self.model(real_x)
        _, e_fake = self.model(fake_x)

        diffs = {}
        for i in range(len(self.channels)):
            diffs[i] = (normalize_tensor(e_real[i]) - normalize_tensor(e_fake[i])) ** 2

        return sum([spatial_average(self.nets[i].model(diffs[i])) for i in range(len(self.channels))])
        
class LossLinLayer(nn.Module):
    """Conv noising layer used in LPIPS loss
       Note: Paper allows for controlling dropout, I'm going to always use it for now"""
    def __init__(self, in_c, out_c=1):
        super(LossLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        )
    
    def forward(self, x):
        return self.model(x)

def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

def normalize_tensor(x):
    factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (factor + 1e-8)
