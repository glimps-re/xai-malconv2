"""
Certainty

Assesses whether the explanation reflects the model’s confidence in its prediction. We used
Shannon entropy to quantify positive contributions from a given method, aiming to minimize entropy for
positive contributions in cases where the model makes confident predictions.
"""

import numpy as np
from scipy.stats import entropy


def compute_certainty(attributions: np.array):
    """
    Compute and return the Shannon entropy for positive attributions in the case of binary classification.
    The entropy is normalized to the range [0, 1] based on the number of positive attributions.
    This version handles multiple attribution sets.

    Parameters
    ----------
    attributions : list or np.array
        A list or array of attribution values (e.g., SHAP or LIME values) for a binary classification task.
        It can be a 2D array (n_samples, n_features) where each row corresponds to the attributions for a single sample.
        Only positive attributions are considered in the calculation.

    Returns
    -------
    tuple
        - H (np.array): The Shannon entropy for each set of attributions.
        - H_max (np.array): The maximum possible entropy for each set, corresponding to a uniform distribution.
        - H_norm (np.array): The normalized Shannon entropy for each set, in the range [0, 1].

    Notes
    -----
    The entropy is computed only for positive attributions, in our use case for malware classification.
    """
    if isinstance(attributions, list):
        attributions = np.array(attributions)

    if attributions.ndim == 1:
        attributions = attributions.reshape(1, -1)

    H = np.zeros(attributions.shape[0])
    H_max = np.zeros(attributions.shape[0])
    H_norm = np.zeros(attributions.shape[0])

    for i, attribution_set in enumerate(attributions):
        positive_attributions = attribution_set[attribution_set > 0]

        if len(positive_attributions) == 0:
            H[i] = 0
            H_max[i] = 0
            H_norm[i] = 0
            continue

        sum_abs_attributions = np.sum(np.abs(positive_attributions))
        normalized_attributions = np.abs(positive_attributions) / sum_abs_attributions

        n = len(normalized_attributions)
        # Maximum possible entropy (log2(n)), for each set
        H_max[i] = np.log2(n)
        # Shannon entropy
        H[i] = entropy(normalized_attributions, base=2)
        # Normalized entropy
        H_norm[i] = (H[i] - 0) / (H_max[i] - 0) if H_max[i] != 0 else 0

    return H, H_max, H_norm
