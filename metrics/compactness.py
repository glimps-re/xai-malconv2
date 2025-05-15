"""
Compactness

Measures the size of an explanation like in `From anecdotal
evidence to quantitative evaluation methods: A systematic review on evaluating
explainable ai.`. It is motivated by human cognitive limitations, as explanations
should be short, sparse, and non-redundant to remain understandable.
A compact explanation relies on a small subset of the most important
features, avoiding overwhelming users with too much information.
"""

import numpy as np


def compute_compactness_with_threshold(
    attributions: np.ndarray, threshold: float = 0.90
):
    """Compute the number of features needed to exceed the given threshold of importance
    for multiple instances of attributions.

    Parameters
    ----------
    attributions : np.ndarray
        Array of shape (n_instances, n_features) containing feature attributions (importance scores).
    threshold : float, optional
        Cumulative importance threshold to reach (default is 0.90).

    Returns
    -------
    np.ndarray
        Array of shape (n_instances,) with the number of features needed for each instance.
    np.ndarray
        Array of shape (n_instances, n_features) with normalized attributions as percentages.
    """
    if attributions.ndim == 1:
        attributions = attributions.reshape(1, -1)

    total_importance = np.sum(np.abs(attributions), axis=1, keepdims=True)
    total_importance[total_importance == 0] = 1
    percentage_feature = np.abs(attributions) / total_importance
    sorted_features = np.sort(percentage_feature, axis=1)[:, ::-1]

    cumsum = np.cumsum(sorted_features, axis=1)

    num_features = np.apply_along_axis(
        lambda x: np.searchsorted(x, threshold) + 1, axis=1, arr=cumsum
    )
    return num_features, percentage_feature
