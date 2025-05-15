"""
Stability

Measures the similarity of explanations for similar instances. High stability indicates that
small variations in the features do not significantly impact the explanation, except in cases where they
strongly alter the prediction. Lack of stability may result from variance in the explainability method or
non-deterministic components.

Source:
    - https://github.com/MAIF/shapash
    - https://towardsdatascience.com/building-confidence-on-explainability-methods-66b9ee575514/
"""

import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP


def _compute_distance(x1, x2, mean_vector, epsilon=0.0000001):
    """
    Compute distances between data points by using L1 on normalized data : sum(abs(x1-x2)/(mean_vector+epsilon))
    Parameters
    ----------
    x1 : array
        First vector
    x2 : array
        Second vector
    mean_vector : array
        Each value of this vector is the std.dev for each feature in dataset
    Returns
    -------
    diff : float
        Returns :math:`\\sum(\\frac{|x1-x2|}{mean\\_vector+epsilon})`
    """
    diff = np.sum(np.abs(x1 - x2) / (mean_vector + epsilon))
    return diff


def _compute_similarities(instance, dataset):
    """
    Compute pairwise distances between an instance and all other data points
    Parameters
    ----------
    instance : 1D array
        Reference data point
    dataset : 2D array
        Entire dataset used to identify neighbors
    Returns
    -------
    similarity_distance : array
        V[j] == distance between actual instance and instance j
    """
    mean_vector = np.array(dataset, dtype=np.float32).std(axis=0)
    similarity_distance = np.zeros(dataset.shape[0])

    for j in range(0, dataset.shape[0]):
        # Calculate distance between point and instance j
        dist = _compute_distance(instance, dataset[j], mean_vector)
        similarity_distance[j] = dist

    return similarity_distance


def _get_radius(dataset, n_neighbors, sample_size=50, percentile=95):
    """
    Calculate the maximum allowed distance between points to be considered as neighbors
    Parameters
    ----------
    dataset : DataFrame
        Pool to sample from and calculate a radius
    n_neighbors : int
        Maximum number of neighbors considered per instance
    sample_size : int, optional
        Number of data points to sample from dataset, by default 500
    percentile : int, optional
        Percentile used to calculate the distance threshold, by default 95
    Returns
    -------
    radius : float
        Distance threshold
    """
    # Select 500 points max to sample
    size = min([dataset.shape[0], sample_size])
    # Randomly sample points from dataset
    sampled_instances = dataset[np.random.randint(0, dataset.shape[0], size), :]
    # Define normalization vector
    mean_vector = np.array(dataset, dtype=np.float32).std(axis=0)
    # Initialize the similarity matrix
    similarity_distance = np.zeros((size, size))
    # Calculate pairwise distance between instances
    for i in range(size):
        for j in range(i, size):
            dist = _compute_distance(
                sampled_instances[i], sampled_instances[j], mean_vector
            )
            similarity_distance[i, j] = dist
            similarity_distance[j, i] = dist
    # Select top n_neighbors
    ordered_X = np.sort(similarity_distance)[:, 1 : n_neighbors + 1]
    # Select the value of the distance that captures XX% of all distances (percentile)
    return np.percentile(ordered_X.flatten(), percentile)


def find_neighbors(selection, dataset, model, mode, n_neighbors=5):
    """
    For each instance, select neighbors based on 3 criteria:
    1. First pick top N closest neighbors (L1 Norm + st. dev normalization)
    2. Filter neighbors whose model output is too different from instance (see condition below)
    3. Filter neighbors whose distance is too big compared to a certain threshold
    Parameters
    ----------
    selection : list
        Indices of rows to be displayed on the stability plot
    dataset : DataFrame
        Entire dataset used to identify neighbors
    model : model object
        ML model
    mode : str
        "classification" or "regression"
    n_neighbors : int, optional
        Top N neighbors initially allowed, by default 10
    Returns
    -------
    all_neighbors : list of 2D arrays
        Wrap all instances with corresponding neighbors in a list with length (#instances).
        Each array has shape (#neighbors, #features) where #neighbors includes the instance itself.
    """
    instances = dataset.loc[selection].values

    all_neighbors = np.empty((0, instances.shape[1] + 1), float)
    """Filter 1 : Pick top N closest neighbors"""
    for instance in instances:
        c = _compute_similarities(instance, dataset.values)
        # Pick indices of the closest neighbors (and include instance itself)
        neighbors_indices = np.argsort(c)[: n_neighbors + 1]
        # Return instance with its neighbors
        neighbors = dataset.values[neighbors_indices]
        # Add distance column
        neighbors = np.append(
            neighbors, c[neighbors_indices].reshape(n_neighbors + 1, 1), axis=1
        )
        all_neighbors = np.append(all_neighbors, neighbors, axis=0)

    # Calculate predictions for all instances and corresponding neighbors
    if mode == "regression":
        # For XGB it is necessary to add columns in df, otherwise columns mismatch
        predictions = model.predict(
            pd.DataFrame(all_neighbors[:, :-1], columns=dataset.columns)
        )
    elif mode == "classification":
        predictions = (
            model(
                torch.tensor(
                    pd.DataFrame(all_neighbors[:, :-1], columns=dataset.columns).values,
                    dtype=torch.float32,
                )
            )
            .squeeze()
            .detach()
            .numpy()
        )

    # Add prediction column
    all_neighbors = np.append(
        all_neighbors, predictions.reshape(all_neighbors.shape[0], 1), axis=1
    )
    # Split back into original chunks (1 chunck = instance + neighbors)
    all_neighbors = np.split(all_neighbors, instances.shape[0])

    """Filter 2 : neighbors with similar blackbox output"""
    # Remove points if prediction is far away from instance prediction
    if mode == "regression":
        # Trick : use enumerate to allow the modifcation directly on the iterator
        for i, neighbors in enumerate(all_neighbors):
            all_neighbors[i] = neighbors[
                abs(neighbors[:, -1] - neighbors[0, -1]) < 0.1 * abs(neighbors[0, -1])
            ]
    elif mode == "classification":
        for i, neighbors in enumerate(all_neighbors):
            all_neighbors[i] = neighbors[abs(neighbors[:, -1] - neighbors[0, -1]) < 0.1]

    """Filter 3 : neighbors below a distance threshold"""
    # Remove points if distance is bigger than radius
    radius = _get_radius(dataset.values, n_neighbors)

    for i, neighbors in enumerate(all_neighbors):
        # -2 indicates the distance column
        all_neighbors[i] = neighbors[neighbors[:, -2] < radius]
    return all_neighbors


def attributions_neighbors(instance, x_encoded, contributions, mode):
    """
    For an instance and corresponding neighbors, calculate various
    metrics (described below) that are useful to evaluate local stability
    Parameters
    ----------
    instance : 2D array
        Instance + neighbours with corresponding features
    x_encoded : DataFrame
        Entire dataset used to identify neighbors
    contributions : DataFrame
        Calculated contribution values for the dataset
    Returns
    -------
    norm_attr_values : array
        Normalized attributions values (with corresponding sign) of instance and its neighbors
    average_diff : array
        Variability (stddev / mean) of normalized attributions (using L1) across neighbors for each feature
    norm_abs_attr_values[0, :] : array
        Normalized absolute attributions of the instance
    """
    # Extract attributions for instance and neighbors
    # :-2 indicates that two columns are disregarded : distance to instance and model output
    ind = (
        pd.merge(
            x_encoded.reset_index(),
            pd.DataFrame(instance[:, :-2], columns=x_encoded.columns),
            how="inner",
        )
        .set_index(
            x_encoded.index.name if x_encoded.index.name is not None else "index"
        )
        .index
    )
    # If classification, select contrbutions of one class only
    if mode == "classification" and len(contributions) == 2:
        contributions = contributions[1]
    attributions = contributions.loc[ind]
    # For neighbors comparison, the sign of attribution values is taken into account
    norm_attr_values = normalize(attributions, axis=1, norm="l1")
    # But not for the average impact of the features across the dataset
    norm_abs_attr_values = normalize(np.abs(attributions), axis=1, norm="l1")
    # Compute the average difference between the instance and its neighbors
    # And replace NaN with 0
    average_diff = np.divide(
        norm_attr_values.std(axis=0),
        norm_abs_attr_values.mean(axis=0),
        out=np.zeros(norm_abs_attr_values.shape[1]),
        where=norm_abs_attr_values.mean(axis=0) != 0,
    )

    return norm_attr_values, average_diff, norm_abs_attr_values[0, :]


def compute_stability(
    x: np.array,
    selection: list,
    contributions: np.array,
    model: Extracted_MLP,
    n_neighbors=5,
    case: str = "classification",
):
    """
    Computes feature stability metrics for a selection of instances.

    - If `selection` contains a single instance, the method returns the normalized contribution
      values of the instance and its corresponding neighbors.
    - If `selection` contains multiple instances, the method returns:
      - The average normalized contribution values across instances and neighbors (amplitude).
      - The variability of these contribution values in the neighborhood (variability).

    Parameters
    ----------
    x : np.ndarray
        Feature matrix where each row represents an instance.
    selection : list of int
        Indices of the instances to be analyzed for stability.
    contributions : np.ndarray
        Attribution scores or feature contributions for the instances.
    model : object
        The predictive model used to generate explanations.

    Returns
    -------
    dict
        If `selection` contains a single instance:
            - "norm_attr" (np.ndarray): Normalized attribution scores for the instance and its neighbors.
        If `selection` contains multiple instances:
            - "amplitude" (np.ndarray): Mean normalized contribution values of instances and neighbors.
            - "variability" (np.ndarray): Variability of contributions within the neighborhood.
    """
    x_encoded = pd.DataFrame(x)
    x_init = pd.DataFrame(x)
    contributions = pd.DataFrame(contributions)

    all_neighbors = find_neighbors(selection, x_encoded, model, case, n_neighbors)

    # Check if entry is a single instance or not
    if len(selection) == 1:
        # Compute explanations for instance and neighbors
        norm_attr, _, _ = attributions_neighbors(
            all_neighbors[0], x_encoded, contributions, case
        )
        local_neighbors = {"norm_attr": norm_attr}
        return local_neighbors
    else:
        numb_expl = len(selection)
        amplitude = np.zeros((numb_expl, x_init.shape[1]))
        variability = np.zeros((numb_expl, x_init.shape[1]))
        # For each instance (+ neighbors), compute explanation
        for i in range(numb_expl):
            (_, variability[i, :], amplitude[i, :]) = attributions_neighbors(
                all_neighbors[i], x_encoded, contributions, case
            )
        features_stability = {"variability": variability, "amplitude": amplitude}
        return features_stability


def compute_determinism(instances, attribution_fn: callable, n=100):
    """
    Compute the average euclidean distance between the attributions of a given method
    for multiple instances over multiple iterations and return the mean determinism across all instances.
    Additionally, measure the average inference time for the attribution function.

    Parameters:
    - instances: A list or array of instances (data points) for which the attributions are computed.
    - attribution_fn (callable): The attribution function to apply to each instance.
      It should return attributions (e.g., Shapley values, feature importance scores).
    - n (int, optional): The number of iterations to compute the attributions for each instance. Default is 100.

    Returns:
    - float: The average of exponential euclidean distance across all instances, which reflects the overall determinism of the attribution method.
    - float: The average inference time per iteration across all instances.
    """
    distance_avg = []
    duration_avg = []

    assert len(instances.shape) > 1

    for instance in instances:
        total_attributions = []
        instance_inference_time = []

        for _i in range(0, n):
            start_time = time.time()
            attributions = attribution_fn(instance.reshape(1, -1))
            end_time = time.time()

            inference_time = end_time - start_time
            instance_inference_time.append(inference_time)

            total_attributions.append(attributions[0])

        euclidean_distance = pdist(total_attributions, metric="euclidean")
        mean_distance = np.mean(euclidean_distance)
        distance = np.exp(-mean_distance)

        distance_avg.append(distance)
        duration_avg.append(np.mean(instance_inference_time))

    mean_determinism = np.mean(distance_avg)
    avg_inference_time = np.mean(duration_avg)

    return mean_determinism, avg_inference_time
