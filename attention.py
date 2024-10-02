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
import torch
from torch import nn


# pylint: disable=too-many-instance-attributes
class PreNormEncoder(nn.Module):
    """Multihead attention encoder w/prenorm residual. """
    def __init__(
        self,
        in_features: int,
        emb_features: int,
        num_heads: int,
        return_attn: bool = False
    ) -> None:
        """
        Args:
            in_features (int): input token dimension
            emb_features (int): embedding dimension
            num_heads (int): number of attention heads
            return_attn (bool, optional): whether to return attention values
        """
        super().__init__()
        self._return_attn = return_attn

        # token embeddings and multihead attention
        self._norm_mha = nn.LayerNorm(in_features)
        self._query_emb = nn.Linear(in_features, emb_features)
        self._key_emb = nn.Linear(in_features, emb_features)
        self._value_emb = nn.Linear(in_features, emb_features)
        self._attn = nn.MultiheadAttention(
            self._query_emb.out_features,
            num_heads=num_heads,
            batch_first=True
        )
        self._proj = nn.Linear(emb_features, in_features)

        # position-wise feed-forward network
        self._norm_ffn = nn.LayerNorm(in_features)
        self._feedfwd = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.LeakyReLU(),
            nn.Linear(4 * in_features, in_features)
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Get new feature tokens.

        Args:
            inp (torch.Tensor): input tokens (M, L, D)

        Returns:
            torch.Tensor: new tokens (M, L, D)
        """

        # prenorm multihead attention
        feats = self._norm_mha(inp)
        query = self._query_emb(feats)
        key = self._key_emb(feats)
        value = self._value_emb(feats)
        feats, attn = self._attn(query, key, value)
        feats = self._proj(feats)
        feats = inp + feats

        # prenorm feed-forward
        feats = self._norm_ffn(feats)
        feats = feats + self._feedfwd(feats)

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
        in_features: int,
        emb_features: int,
        num_heads: int,
        num_classes: int,
    ) -> None:
        """
        Args:
            in_features (int): input token dimension
            emb_features (int): embedding dimension
            num_heads (int): number of attention heads
            num_classes (int): number of possible target classes
            return_attn (bool, optional): whether to return attention values
        """
        super().__init__()
        self._feature_extractor = PreNormEncoder(
            in_features,
            emb_features,
            num_heads,
            return_attn=False
        )
        self._cls = nn.Linear(in_features, num_classes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Classify sequences.

        Args:
            inp (torch.Tensor): sequences to classify (M, L, D)

        Returns:
            torch.Tensor: classification logits (M, K)
        """
        feats = self._feature_extractor(inp)
        feats = feats[:, 0, :]  # only need 1st token
        return self._cls(feats)
