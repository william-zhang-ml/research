"""Validation shortcuts. """
from typing import Dict, Tuple
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_score,
    PrecisionRecallDisplay,
    recall_score,
    RocCurveDisplay
)


def add_key_prefix(data: Dict, prefix: str) -> None:
    """Add prefix to dictionary keys IN PLACE.

    Args:
        data (Dict): dictionary to modify
        prefix (str): prefix to prepend to keys
    """
    keys = list(data.keys())
    for key in keys:
        data[f'{prefix}{key}'] = data[key]
    for key in keys:
        del data[key]


def add_key_suffix(data: Dict, suffix: str) -> None:
    """Add suffix to dictionary keys IN PLACE.

    Args:
        data (Dict): dictionary to modify
        suffix (str): suffix to prepend to keys
    """
    keys = list(data.keys())
    for key in keys:
        data[f'{key}{suffix}'] = data[key]
    for key in keys:
        del data[key]


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
    for batch_imgs, batch_labels in tqdm(loader, leave=False):
        labels.append(batch_labels)
        scores.append(model(batch_imgs.to(device)).softmax(dim=1).cpu())
        if isinstance(num_samp, int):
            if len(labels) * len(labels[0]) >= num_samp:
                break
    labels = torch.cat(labels).cpu().numpy()
    scores = torch.cat(scores).cpu().numpy()
    return {
        'labels': labels,
        'scores': scores,
        'pred': scores.argmax(axis=1),
        'binary_labels': labels == 0,
        'binary_scores':  scores[:, 0]
    }


def get_cls_metrics(outputs: Dict) -> Dict:
    """Compute metrics from get_cls_outputs() outputs.

    Args:
        outputs (Dict): output from get_cls_outputs()

    Returns:
        Dict: classification metrics
    """
    labels, pred, binary_labels, binary_scores = (
        outputs['labels'],
        outputs['pred'],
        outputs['binary_labels'],
        outputs['binary_scores']
    )
    return {
        'top1': (labels == pred).mean(),
        'binary-prec': precision_score(binary_labels, binary_scores > 0.5),
        'binary-recall': recall_score(binary_labels, binary_scores > 0.5)
    }


def get_cls_displays(outputs: Dict) -> Dict:
    """Draw graphs from get_cls_outputs() outputs.

    Args:
        outputs (Dict): output from get_cls_outputs()

    Returns:
        Dict: classification displays
    """
    labels, pred, binary_labels, binary_scores = (
        outputs['labels'],
        outputs['pred'],
        outputs['binary_labels'],
        outputs['binary_scores']
    )
    displays = {
        'confmat': ConfusionMatrixDisplay.from_predictions(
            labels, pred,
            colorbar=False,
            cmap='Blues',
            include_values=max(labels) < 10,
            normalize='true'
        ),
        'confmat2': ConfusionMatrixDisplay.from_predictions(
            labels, pred,
            colorbar=False,
            cmap='Blues',
            include_values=max(labels) < 10,
            normalize=None
        ),
        'binary-confmat': ConfusionMatrixDisplay.from_predictions(
            binary_labels, binary_scores > 0.5,
            colorbar=False,
            cmap='Blues',
            include_values=True,
            normalize='true'
        ),
        'binary-confmat2': ConfusionMatrixDisplay.from_predictions(
            binary_labels, binary_scores > 0.5,
            colorbar=False,
            cmap='Blues',
            include_values=True,
            normalize=None
        ),
        'roc': RocCurveDisplay.from_predictions(binary_labels, binary_scores),
        'pr': PrecisionRecallDisplay.from_predictions(
            binary_labels,
            binary_scores
        )
    }
    displays['roc'].ax_.grid()
    displays['pr'].ax_.grid()
    return displays


def run_cls_eval(
    model,
    loader,
    device: str = 'cuda:0',
    num_samp: int = None,
    tag: str = None
) -> Tuple[Dict, ...]:
    """Run classification evaluation pipeline: inference -> metrics + graphs.

    Args:
        model (Module): model to run data through
        loader (DataLoader): batch generator
        device (str): device the model is on
        num_samp (int): number of samples before exiting early
        tag (str): tag to prepend to keys

    Returns:
        Tuple[Dict, ...]: classification outputs, metrics, displays
    """
    outputs = get_cls_outputs(model, loader, device, num_samp)
    metrics = get_cls_metrics(outputs)
    displays = get_cls_displays(outputs)
    if isinstance(tag, str):
        if tag[-1] != '-':
            tag = tag + '-'
        for data in [outputs, metrics, displays]:
            add_key_prefix(data, tag)
    return outputs, metrics, displays
