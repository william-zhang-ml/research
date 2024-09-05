"""Miscellaneous AI/ML code. """
from copy import deepcopy
from typing import Union
import torch
from torch import nn
from torch.autograd import Function
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


def clip_loss(
    a_embeddings: torch.Tensor,
    b_embeddings: torch.Tensor,
    temp: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Compute cross-domain contrastive cross-entropy.

    Args:
        a_embeddings (torch.Tensor): embedding vectors from domain A
        b_embeddings (torch.Tensor): embedding vectors from domain B
        temp (Union[float, torch.Tensor]): log temperature (reciprocal)
    """
    device = a_embeddings.device

    # compute cosine similarity
    cos_sim = torch.einsum(
        'id,jd->ij',
        normalize(a_embeddings, p=2, dim=1),
        normalize(b_embeddings, p=2, dim=1),
    )

    # cross-sample cross-entropy
    assert cos_sim.shape[0] == cos_sim.shape[1]
    labels = torch.arange(a_embeddings.shape[0], device=device)
    if temp:
        if isinstance(temp, float):
            temp = torch.tensor(float, device=device)
        logits = cos_sim * temp.exp()
    else:
        logits = cos_sim
    loss_1 = cross_entropy(logits, labels)
    loss_2 = cross_entropy(logits.T, labels)
    return (loss_1 + loss_2) / 2


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


class GradientReversalNode(Function):
    """Utility that reverses gradients during backpropagation. """
    @staticmethod
    def forward(_, inp: torch.Tensor) -> torch.Tensor:
        """No-op function (required by API).

        Args:
            inp (torch.Tensor): some tensor

        Returns:
            torch.Tensor: unchanged tensor
        """
        return inp

    @staticmethod
    def backward(_, gradient: torch.Tensor) -> torch.Tensor:
        """Reverse gradients.

        Args:
            gradient (torch.Tensor): some tensor's gradient

        Returns:
            torch.Tensor: negative gradient
        """
        return gradient.neg()


class StudentTeacherAverager:
    """Utility for updating a model that is a moving average of another. """
    def __init__(
        self,
        source: nn.Module,
        average: nn.Module,
        momentum: float
    ) -> None:
        """
        Args:
            source (nn.Module): model being trained (ws)
            average (nn.Module): running weight average (wa)
            momentum (float): amount current average counts in new average (m)
        """
        self._source = source
        self._average = average
        self._momentum = momentum

    def step(self) -> None:
        """Apply moving average weight update: wa = m * wa + (1 - m) * ws. """
        average_state = self._average.state_dict()  # references, not copies
        for param, value in self._source.state_dict().items():
            average_state[param] *= self._momentum
            average_state[param] += (1 - self._momentum) * value
