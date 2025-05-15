"""
Accuracy

Quantifies how well the explanation aligns with the model’s prediction. This is done by
summing the attributions of the explainability methods for a given instance and comparing this sum to
the model’s initial prediction, using classical Accuracy score. The model initial prediction becoming our
target and the prediction based on explainability our new prediction in thatscenario.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def compute_accuracy(attributions, y_preds, additivity=False, reference=None):
    """
    Compute the accuracy between the sum of feature attributions and the model's binary predictions.

    This function calculates the sum of feature attributions for each instance and compares it to the
    model's predicted class (binary classification) to determine the accuracy. The attributions are
    summed up, and if the sum is non-negative, it's classified as class 1, otherwise class 0. The same
    is done for the model predictions, where a threshold of 0.5 is applied to the predicted probabilities.

    Parameters
    ----------
    attributions : array-like, shape (n_samples, n_features)
        The attribution scores for each feature and instance, usually obtained from an explainability method
        such as SHAP, LIME, etc. Each row corresponds to an instance, and each column corresponds to a feature.

    y_preds : array-like, shape (n_samples,)
        The predicted probabilities or binary outputs (0 or 1) from the model. If the values are probabilities,
        they are thresholded at 0.5 to determine the class label.

    Returns
    -------
    float
        The accuracy score between the summed attributions and the model's predictions. The score ranges from
        0 (no match) to 1 (perfect match).
    """
    if additivity:
        if reference is None:
            raise ValueError("Please, provide a reference to compute additivity.")
        y_probs = (
            np.sum(attributions, axis=1) + reference
        )  # adjust attributions based on reference
        y_probs = np.where(y_probs >= 0.5, 1, 0)
    else:
        y_probs = np.sum(attributions, axis=1)  # default: sum of attributions
        y_probs = np.where(y_probs >= 0, 1, 0)
    y_preds = np.where(np.array(y_preds) >= 0.5, 1, 0)
    return accuracy_score(y_true=y_preds, y_pred=y_probs)
