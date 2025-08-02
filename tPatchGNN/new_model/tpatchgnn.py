import torch
import torch.nn as nn

from new_model.encoder import IMTSEncoder
from new_model.decoder import ForecastDecoder


class tPatchGNN(nn.Module):
    """
    High-level model wrapper that combines:
        - Encoder (TTCN + Transformer + GCN/CrossAttention)
        - Decoder (forecasting head)
    """

    def __init__(self, args, supports=None):
        super().__init__()
        self.encoder = IMTSEncoder(args, supports)
        self.decoder = ForecastDecoder(args)
        self.args = args

    def forward(self, X, mask_X, te_pred, supports=None):
        """
        Args:
            X: (B*N*M, L, F) patched input with features
            mask_X: (B*N*M, L, 1) mask for missing values
            te_pred: (B, N, Lp, F_te) time embeddings for prediction
            supports: adjacency matrices (optional)

        Returns:
            outputs: (B, N, Lp) forecasts
        """
        # Encode input
        h = self.encoder(X, mask_X, supports)  # (B, N, hid_dim)

        # Repeat across prediction horizon
        B, N, D = h.shape
        Lp = te_pred.shape[2]
        h = h.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, N, Lp, D)

        # Decode
        outputs = self.decoder(h, te_pred).squeeze(-1)  # (B, N, Lp)
        return outputs
