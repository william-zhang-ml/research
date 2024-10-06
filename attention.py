"""
Attention and transformer implementations.

REFERENCES

Attention is All You Need
- original concept

On Layer Normalization in the Transformer Architecture
- argued for prenorm residuals instead of postnorm

An Image iS Worth 16x16 Words: Transformers for Image Recognition at Scale
- connected a perceptron to the first output token for classification
"""
from typing import Sequence, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


def get_key_padding_mask(lengths: Sequence[int]) -> torch.Tensor:
    """Get a mask that indicates which key tokens are padding.

    Args:
        lengths (Sequence[int]): lengths of key sequences

    Returns:
        torch.Tensor: key padding mask
    """
    max_length = max(lengths)
    mask = torch.zeros(len(lengths), max_length)
    for i_seq, curr in enumerate(lengths):
        mask[i_seq, curr:] = -torch.inf
    return mask


def pack_and_pad(
    tokens: Sequence[torch.Tensor],
    enforce_sorted: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Utility for packing token sequences.

    Args:
        tokens (Sequence[torch.Tensor]): tokens sequences to pack
        enforce_sorted (bool): whether sequences need to be length-ordered

    Return:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            padded token sequences, sequence lengths, padding mask
    """
    packed = pack_sequence(tokens, enforce_sorted=enforce_sorted)
    padded, lengths = pad_packed_sequence(packed, batch_first=True)
    mask = get_key_padding_mask(lengths)
    return padded, lengths, mask


# pylint: disable=too-many-instance-attributes
class PreNormEncoder(nn.Module):
    """Multihead attention encoder w/prenorm residual. """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        return_attn: bool = False
    ) -> None:
        """
        Args:
            embed_dim (int): embedding dimension
            num_heads (int): number of attention heads
            return_attn (bool, optional): whether to return attention values
        """
        super().__init__()
        self._return_attn = return_attn

        # token embeddings and multihead attention
        self._norm_mha = nn.LayerNorm(embed_dim)
        self._attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # position-wise feed-forward network
        self._norm_ffn = nn.LayerNorm(embed_dim)
        self._feedfwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(
        self,
        inp: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Get new feature tokens.

        Args:
            inp (torch.Tensor): input tokens (M, L, D)
            key_padding_mask (torch.Tensor): which tokens are padding (M, L)

        Returns:
            torch.Tensor: new tokens (M, L, D)
        """

        # prenorm multihead attention
        residual = self._norm_mha(inp)
        residual, attn = self._attn(
            residual, residual, residual,
            key_padding_mask=key_padding_mask
        )
        feats = inp + residual

        # prenorm feed-forward
        residual = self._norm_ffn(feats)
        residual = self._feedfwd(residual)
        feats = feats + residual

        if self._return_attn:
            outp = feats, attn
        else:
            outp = feats
        return outp
# pylint: enable=too-many-instance-attributes


class SequenceClassifier(nn.Module):
    """Attention-based classification head. """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
    ) -> None:
        """
        Args:
            embed_dim (int): embedding dimension
            num_heads (int): number of attention heads
            num_classes (int): number of possible target classes
            return_attn (bool, optional): whether to return attention values
        """
        super().__init__()
        self._feature_extractor = PreNormEncoder(
            embed_dim,
            num_heads,
            return_attn=False
        )
        self._cls = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        inp: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Classify sequences.

        Args:
            inp (torch.Tensor): sequences to classify (M, L, D)
            key_padding_mask (torch.Tensor): which tokens are padding (M, L)

        Returns:
            torch.Tensor: classification logits (M, K)
        """
        feats = self._feature_extractor(inp, key_padding_mask)
        feats = feats[:, 0, :]  # only need 1st token
        return self._cls(feats)
