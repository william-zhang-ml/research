"""
This module implements "SCAN: Learning to Classify Images without Labels",
which draws from
"Unsupervised Feature Learning via Non-Parametric Instance Discrimination".
"""
import torch
from torch.nn.functional import normalize


def nonparameteric_softmax_loss(
    emb: torch.Tensor,
    temperature: float = 1.
) -> torch.Tensor:
    """Compute nonparametric softmax loss.

    Args:
        emb (torch.Tensor): image embeddings (B, D)
        temperature (float): logit scaling coefficient
    
    Returns:
        torch.Tensor:
    """
    emb = normalize(emb, p=2, dim=-1)
    logits = (emb @ emb.transpose(-2, -1)) / temperature
    return logits.softmax(dim=-1).diag().log().sum().neg()


def scan_loss(
    logits: torch.Tensor,
    neighbor_logits: torch.Tensor,
    reg: float = 1
) -> torch.Tensor:
    """Compute nearest neighbor consistency loss.

    Args:
        logits (torch.Tensor): image classification logits
        neighbor_logits (torch.Tensor): image neighbors classification logits
        reg (float, optional): entropy regularization amount. Defaults to 1.

    Returns:
        torch.Tensor: nearest neighbor consistency loss
    """
    softmax = logits.softmax(dim=-1)
    neighbor_softmax = neighbor_logits.softmax(dim=-1)

    # consistency encourages same labels in a cluster
    consistency = torch.bmm(
        softmax.unsqueeze(-2),
        neighbor_softmax.transpose(-2, -1)
    ).squeeze()
    consistency = consistency.log().mean().neg()

    # entropy regularization prevents classifier collapse
    neg_entropy = softmax.mean(dim=0)
    neg_entropy = neg_entropy * neg_entropy.log()
    neg_entropy = neg_entropy.sum()

    return consistency + reg * neg_entropy
