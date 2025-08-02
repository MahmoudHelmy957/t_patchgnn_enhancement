import torch
import torch.nn as nn
import torch.nn.functional as F

from new_model.layers.transformer import PositionalEncoding, build_transformer_encoder
from new_model.layers.graph import GCNLayer
from new_model.layers.ttcn import TTCN


class IMTSEncoder(nn.Module):
    """
    Irregular Multivariate Time Series Encoder
    Pipeline:
        TTCN (patch encoding) →
        Transformer (intra-channel temporal modeling) →
        GCN (inter-channel modeling)
    """

    def __init__(self, args, supports=None):
        super().__init__()

        self.hid_dim = args.hid_dim
        self.nlayer = args.nlayer
        self.nhead = args.nhead
        self.tf_layer = args.tf_layer
        self.dropout = args.dropout
        self.hop = args.hop
        self.te_dim = args.te_dim

        # Patch encoder (TTCN)
        self.ttcn = TTCN(input_dim=1 + args.te_dim, hid_dim=args.hid_dim)

        # Transformer encoder (intra-channel temporal modeling)
        self.pos_enc = PositionalEncoding(args.hid_dim)
        self.transformers = nn.ModuleList([
            build_transformer_encoder(args.hid_dim, args.nhead, args.tf_layer)
            for _ in range(args.nlayer)
        ])

        # Inter-channel modeling block (GCN)
        support_len = (len(supports) if supports is not None else 1)
        self.inter_channel_blocks = nn.ModuleList([
            GCNLayer(args.hid_dim, args.hid_dim,
                     dropout=args.dropout,
                     support_len=support_len,
                     order=args.hop)
            for _ in range(args.nlayer)
        ])

        self.supports = supports if supports is not None else []

    def forward(self, X_int, mask_X, supports=None):
        """
        Args:
            X_int: (B*N*M, L, F) patched input with features (values + time embeddings)
            mask_X: (B*N*M, L, 1) mask for missing values
            supports: list of adjacency matrices (optional dynamic supports)

        Returns:
            h: (B, N, hid_dim) encoded representation
        """
        BNM, L, F = X_int.shape
        device = X_int.device

        # 1. Patch encoding (TTCN)
        h_patch = self.ttcn(X_int, mask_X)  # (B*N*M, hid_dim-1)
        mask_patch = (mask_X.sum(dim=1) > 0).float()  # (B*N*M, 1)
        h_patch = torch.cat([h_patch, mask_patch], dim=-1)  # (B*N*M, hid_dim)

        # Reshape to (B, N, M, hid_dim)
        # NOTE: you must pass B, N, M externally (or infer from args)
        # For now, assume args has N, M
        B = int(BNM / (self.args.ndim * self.args.npatch))
        h_patch = h_patch.view(B, self.args.ndim, self.args.npatch, -1)  # (B, N, M, D)

        x = h_patch
        for layer in range(self.nlayer):
            # 2. Transformer (intra-channel)
            B, N, M, D = x.shape
            x_reshaped = x.view(B * N, M, D)  # (B*N, M, D)
            x_reshaped = self.pos_enc(x_reshaped)
            x_reshaped = self.transformers[layer](x_reshaped)  # (B*N, M, D)
            x = x_reshaped.view(B, N, M, D)

            # 3. Inter-channel GCN
            # Permute to (B, F, N, M) for GCN
            x_gcn_in = x.permute(0, 3, 1, 2)  # (B, D, N, M)
            supports_all = self.supports + (supports if supports else [])
            x = self.inter_channel_blocks[layer](x_gcn_in, supports_all)  # (B, D, N, M)

            # Permute back
            x = x.permute(0, 2, 3, 1)  # (B, N, M, D)

        # Aggregate over patches (M dimension) to get final channel representation
        h = x.mean(dim=2)  # (B, N, D)

        return h
