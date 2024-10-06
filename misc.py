"""Miscellaneous research code. """
from typing import Callable, Dict, Sequence, Tuple, Union
import git
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torchvision.ops import box_convert
from tqdm.auto import tqdm


def get_repo_hash() -> str:
    """
    Returns:
        str: long hash of the repo
    """
    repo = git.Repo(search_parent_directories=True)
    return repo.head.commit.hexsha


def bootstrap(
    data: np.ndarray,
    statistic: Callable,
    sample_size: int,
    num_trials: int = 1000
) -> np.ndarray:
    """Compute a statistic multiple times using bootstrap samples.

    Args:
        data (np.ndarray): dataset to sample
        statistic (Callable): statistic to compute
        sample_size (int): number of samples in each bootstrap sampmle
        num_trials (int, optional): number of bootstrap samples to process

    Returns:
        np.ndarray: statistic of interest for each bootstrap sample
    """
    stats = []
    for _ in tqdm(range(num_trials), leave=False):
        sample = np.random.choice(data, size=sample_size, replace=True)
        stats.append(statistic(sample))
    return np.array(stats)


class EmpiricalDistribution:
    """Sample-defined distribution utility. """
    def __init__(self, samples: Sequence[float]) -> None:
        if isinstance(samples, np.ndarray):
            self._samples = np.copy(samples)
        else:
            self._samples = np.array(samples)
        self._samples.sort()

    def __len__(self) -> int:
        return len(self._samples)

    def cdf(
        self,
        query: Union[float, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Compute empirical cumulative distribution function.

        Args:
            query (Union[float, Sequence[float]]): points at which to evaluate

        Returns:
            Union[float, np.ndarray]: cumulative distribution values
        """
        idcs = np.searchsorted(self._samples, query, 'right')
        return idcs / len(self._samples)


def get_pca_contour(data: np.ndarray, stdev: float = 3) -> np.ndarray:
    """Estimate a data-aligned/centered contour using SVD/PCA/whitening math.

    Args:
        data (np.ndarray): enough samples to model distribution stats (N, 2)
        stdev (float): contour value in units of standard deviation

    Returns:
        np.ndarray: contour (300, 2)
    """
    # compute needed data statistics
    samp_mean = data.mean(axis=0)
    [_, sing_val, basis] = np.linalg.svd(data - samp_mean)

    # apply data statistics to a circle with <stdev> radius
    radians = np.linspace(0, 2 * np.pi, 300)
    circle = stdev * np.stack([np.cos(radians), np.sin(radians)], axis=1)
    scaled_basis = np.diag(sing_val / np.sqrt(len(data))) @ basis
    return circle @ scaled_basis + samp_mean


def gen_confmat(
    y_true: Sequence,
    y_pred: Sequence
) -> Tuple[ConfusionMatrixDisplay, float]:
    """Compute normalized confusion matrix w/prettier formatting.

    Args:
        y_true (Sequence): true labels
        y_pred (Sequence): predicted labels

    Returns:
        Tuple[ConfusionMatrixDisplay, float]: confusion matrix, top-1 accuracy
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        cmap='Blues',
        colorbar=False,
        normalize='true'
    )
    top1 = 100 * (y_true == y_pred).float().mean()
    disp.ax_.set_title(f'Top-1 {top1: .1f}%')
    disp.figure_.tight_layout()
    return disp, top1


def gen_tsne_plot(
    data: np.ndarray,
    labels: Sequence[int] = None,
    color: str = 'k',
    cmap: str = 'tab10',
    tsne_kwargs: Dict = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Project a dataset with t-SNE algorithm and plot.

    Args:
        data (np.ndarray): dataset to plot (num_samples, num_features)
        labels (Sequence[int]): sample classification labels
        color (str): marker color if no labels given
        cmap (str): marker colormap if labels given
        tsne_kwargs (Dict): sklearn.manifold.TSNE settings

    Returns:
        Tuple[plt.Figure, plt.Axes]: plot handles
    """
    tsne_kwargs = {} if tsne_kwargs is None else tsne_kwargs
    scatter_kwargs = {
        's': 30,
        'c': color if labels is None else labels,
        'marker': '.',
        'cmap': None if labels is None else cmap
    }

    # project and plot
    proj = TSNE(**tsne_kwargs).fit_transform(data)
    fig, axes = plt.subplots()
    axes.grid()
    axes.scatter(*proj.T, **scatter_kwargs)
    fig.tight_layout()

    return fig, axes


def add_boxes(
    axes: plt.Axes,
    boxes: torch.Tensor,
    fmt: str = 'xyxy',
    show_idx: bool = True,
    margin: float = 0
) -> None:
    """Add boxes to an axes.

    Args:
        axes (plt.Axes): axes to edit
        boxes (torch.Tensor): boxes to add
        fmt (str): box coordinate convention
        show_idx (bool): whether to annotate boxes with their array index
        margin (float): array index text offset
    """
    boxes = box_convert(boxes, fmt, 'xywh')
    for i_box, box in enumerate(boxes):
        patch = Rectangle(box[:2], box[2], box[3], color='r', fill=False)
        axes.add_patch(patch)
        if show_idx:
            axes.text(box[0] + margin, box[1] + margin, i_box, color='r')
