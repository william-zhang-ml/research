"""Validation shortcuts. """
from typing import Dict
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def get_cls_outputs(
    model: Module,
    loader: DataLoader,
    device: str = 'cuda:0',
    num_samp: int = None
) -> Dict:
    """Get classification outputs.

    Args:
        model (Module): model to run data through
        loader (DataLoader): batch generator
        device (str): device the model is on
        num_samp (int): number of samples before exiting early

    Returns:
        Dict: classification outputs
    """
    labels, scores = [], []
    for batch_imgs, batch_labels in tqdm(loader):
        labels.append(batch_labels)
        scores.append(model(batch_imgs.to(device)))
        if isinstance(num_samp, int):
            if len(labels) * len(labels[0]) > num_samp:
                break
    labels = torch.cat(labels).cpu().numpy()
    scores = torch.cat(scores).cpu().numpy()
    return {
        'labels': labels,
        'scores': scores,
        'pred': scores.argmax(axis=1)
    }
