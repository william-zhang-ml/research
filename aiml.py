"""Miscellaneous AI/ML code. """
from copy import deepcopy
import torch
from torch import nn
from torch.nn.functional import cross_entropy, normalize


def count_params(module: nn.Module) -> int:
    """Count the number of trainable module parameters.

    Args:
        module (nn.Module): module of interest

    Returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def jitter_conv2d_weights(
    module: nn.Module,
    stdev: float = 1e-3,
    inplace: bool = False
) -> nn.Module:
    """Recursively add Gaussian noise to all Conv2d kernels in a module.

    Args:
        module (nn.Module): module of which to jtter weights
        stdev (float): noise standard deviation. Defaults to 1e-3.
        inplace (bool): whether to jitter inplace. Defaults to False.

    Returns:
        nn.Module: module with jittered weights
    """
    if not inplace:
        module = deepcopy(module)
    for submodule in module.children():
        if isinstance(submodule, nn.Conv2d):
            submodule.weight.data.add_(
               stdev * torch.randn(submodule.weight.shape)
            )
        else:
            jitter_conv2d_weights(submodule, stdev, True)
    return module


def cross_entropy_w_pseudolabels(
    softlabels: torch.Tensor,
    logits: torch.Tensor,
    thresh: float = 0.,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute cross entropy using pseudolabels.

    Args:
        softlabels (torch.Tensor): batch of sum-to-1 soft labels
        logits (torch.Tensor): logits to evaluate
        thresh (float): soft label score required to use a sample
        reduction (str): reduction to apply to output
    """
    pseudoscores, pseudolabels = softlabels.max(dim=-1)
    use = pseudoscores > thresh
    loss = cross_entropy(logits[use], pseudolabels[use], reduction=reduction)
    return loss


def minimum_class_confusion(
    logits: torch.Tensor,
    temperature: float = 1.
) -> torch.Tensor:
    """Compute minimum class confusion loss.

    Args:
        logits (torch.Tensor): classification logits
        temperature (float): softmax scaling value

    Returns:
        torch.Tensor: class confusion loss
    """
    assert logits.ndim == 2
    scores = logits.div(temperature).softmax(dim=1)
    certainty = scores.mul(scores.log()).sum(dim=1, keepdim=True)
    weight = certainty.add(1).softmax(dim=1).mul(logits.shape[0])
    confusion = weight.mul(scores).T.matmul(scores)
    confusion = normalize(confusion, p=1, dim=1)
    loss = confusion.sum().sub(confusion.trace()).div(logits.shape[1])
    return loss
