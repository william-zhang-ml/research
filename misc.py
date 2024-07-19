"""Miscellaneous research code. """
from typing import Callable
import numpy as np
from tqdm.auto import tqdm


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
