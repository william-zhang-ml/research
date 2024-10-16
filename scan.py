import torch


def scan_loss(
    logits: torch.Tensor,
    neighbor_logits: torch.Tensor,
    reg: float = 1
) -> torch.Tensor:
    softmax = logits.softmax(dim=-1)
    neighbor_softmax = neighbor_logits.softmax(dim=-1)

    # consistency encourages same labels in a cluster
    consistency = softmax.unsqueeze(-2)
    consistency = softmax @ neighbor_softmax.transpose(-2, -1)
    print(consistency.shape)
    print(softmax.unsqueeze(-2).shape)

    # entropy regularization prevents classifier collapse
    entropy = softmax.mean(dim=0)
    entropy = entropy * entropy.log()
    entropy = entropy.sum()

    return reg * entropy
