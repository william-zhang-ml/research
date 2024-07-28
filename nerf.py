"""Miscellaneous research code. """
import torch


def apply_positional_encoding(
    coord: torch.Tensor,
    enc_dim: int
) -> torch.Tensor:
    """Apply positional encoding.

    Args:
        coord (torch.Tensor): coordinates to encode (B, 6)
        enc_dim (int): encoding dimension (L)

    Returns:
        torch.Tensor: encoded coordinates (B, 6, 2L)
    """
    # this code implements equation 4
    freq = (2 ** torch.arange(enc_dim)) * torch.pi
    phase = torch.matmul(
        coord.unsqueeze(-1),      # (B, 6, 1)
        freq.view(1, 1, enc_dim)  # (1, 1, L)
    )  # (B, 6, L)
    return torch.cat([phase.sin(), phase.cos()], dim=-1)
