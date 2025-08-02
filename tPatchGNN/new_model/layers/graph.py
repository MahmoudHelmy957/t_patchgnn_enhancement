import torch
import torch.nn as nn
import torch.nn.functional as F

class NConv(nn.Module):
    """Basic graph convolution using adjacency matrix."""
    def forward(self, x, A):
        # x: (B, F, N, M), A: (B, M, N, N)
        return torch.einsum('bfnm,bmnv->bfvm', x, A).contiguous()

class GraphLinear(nn.Module):
    """1x1 Conv linear layer for graph features."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1,1))

    def forward(self, x):
        return self.mlp(x)

class GCNLayer(nn.Module):
    """GCN layer with multiple adjacency supports."""
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = NConv()
        self.mlp = GraphLinear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a)
                out.append(x1)
        h = torch.cat(out, dim=1)
        return F.relu(self.mlp(h))
