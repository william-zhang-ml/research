"""Miscellaneous AI/ML code. """
from copy import deepcopy
import torch
from torch import nn


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
