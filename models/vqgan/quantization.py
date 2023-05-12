import torch
import torch.nn as nn


class Codebook(nn.Module):
    """Quantization module, inspired by youtuber outlier"""
    def __init__(self, codebook_size, latent_dim, beta):
        super(Codebook, self).__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)
        # initialize embedding weights
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        # Euclidean distance between z_flattened and embedding
        # sum (a-b)^2 = sum(a^2) + sum(b^2) - sum(2*a*b) 
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))
        
        # quantize
        min_encoding_idx = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_idx).view(z.shape)

        # loss := commitment loss + embedding optimization
        loss = torch.mean((z_q.detach() - z)**2) + \
                self.beta * torch.mean((z_q - z.detach())**2)
        
        # gradient preservation funkiness
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_idx, loss
    

