import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    This module injects information about the relative or absolute
    position of tokens in a sequence, so that the model can make
    use of the order of the sequence.

    Args:
        d_model (int): The dimension of the embeddings/model.
        max_len (int, optional): Maximum sequence length to support.
            Default is 512.

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)

    Example::
        >>> pe = PositionalEncoding(d_model=512)
        >>> x = torch.zeros(32, 100, 512)  # (batch, seq_len, d_model)
        >>> out = pe(x)
        >>> out.shape
        torch.Size([32, 100, 512])
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Encoded tensor of same shape as input.
        """
        return x + self.pe[:, :x.size(1), :]


def build_transformer_encoder(d_model: int, nhead: int, num_layers: int) -> nn.TransformerEncoder:
    """
    Builds a Transformer encoder stack using PyTorch's built-in implementation.

    Args:
        d_model (int): The dimension of the embeddings/model.
        nhead (int): The number of attention heads in each layer.
        num_layers (int): Number of encoder layers to stack.

    Returns:
        nn.TransformerEncoder: A Transformer encoder consisting of
        `num_layers` stacked encoder layers.

    Example::
        >>> encoder = build_transformer_encoder(d_model=512, nhead=8, num_layers=6)
        >>> x = torch.zeros(32, 100, 512)  # (batch, seq_len, d_model)
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([32, 100, 512])
    """
    layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=num_layers)
