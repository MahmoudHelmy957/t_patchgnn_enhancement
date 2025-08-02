import torch
import torch.nn as nn
import torch.nn.functional as F

class TTCN(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.ttcn_dim = hid_dim - 1
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, self.ttcn_dim),
            nn.ReLU(),
            nn.Linear(self.ttcn_dim, self.ttcn_dim),
            nn.ReLU(),
            nn.Linear(self.ttcn_dim, input_dim * self.ttcn_dim)
        )
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))

    def forward(self, X_int, mask_X):
        # X_int: (B*N*M, L, F), mask_X: (B*N*M, L, 1)
        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1)
        X_int_broad = X_int.unsqueeze(-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1)
        return torch.relu(ttcn_out + self.T_bias)
