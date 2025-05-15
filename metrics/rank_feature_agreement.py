import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr


def intersection(r1, r2):
    return list(set(r1) & set(r2))


def check_size(r1, r2):
    assert len(r1) == len(r2), "Both rankings should be the same size"


def feature_agreement(r1, r2):
    """
    Measures the fraction of common features between the
    sets of top-k features of the two rankings.

    From Krishna et al. (2022), The Disagreement Problem in
    Explainable Machine Learning: A Practitioner’s Perspective

    Parameters
    ---------------
    r1, r2 : list
        Two feature rankings of identical shape
    """
    check_size(r1, r2)
    k = len(r1)

    return len(intersection(r1, r2)) / k


def rank_agreement(r1, r2):
    """
    Stricter than feature agreement, rank agreement checks
    that the feature order is comparable between the two rankings.

    From Krishna et al. (2022), The Disagreement Problem in
    Explainable Machine Learning: A Practitioner’s Perspective

    Parameters
    ---------------
    r1, r2 : list
        Two feature rankings of identical shape
    """
    check_size(r1, r2)
    k = len(r1)

    return np.sum([True if x == y else False for x, y in zip(r1, r2)]) / k


def weak_rank_agreement(r1, r2):
    """
    Check if the rank is approximately close (within one rank).
    """
    check_size(r1, r2)
    k = len(r1)
    window_size = 1

    rank_agree = []
    for i, v in enumerate(r1):
        if i == 0:
            if v in r2[i : i + window_size + 1]:
                rank_agree.append(True)
            else:
                rank_agree.append(False)
        else:
            if v in r2[i - window_size : i + window_size + 1]:
                rank_agree.append(True)
            else:
                rank_agree.append(False)

    return np.sum(rank_agree) / k


def rank_correlation(r1, r2):
    return spearmanr(r1, r2)


def to_rankings(df, instance):
    """
    Convert feature attributions to a list of top features.
    """
    contrib_features = df.columns

    vals = df[contrib_features].values[instance, :]
    rankings = np.argsort(np.absolute(vals))[::-1]

    return rankings


def compute_matrices(weights, instance, methods, n=10):
    """
    Compute matrices for feature agreement and rank agreement for a specific instance.

    This function calculates the agreement between feature rankings obtained
    from different explainability methods for a single data instance.
    Two types of agreement are computed:
      - **Feature Agreement**: Measures the overlap between the top-k features of the rankings.
      - **Rank Agreement**: Measures the order consistency of features between the rankings.

    Parameters:
        weights (list of pd.DataFrame): A list of dataframes where each dataframe corresponds
            to feature attributions (weights) computed by a specific explainability method.
            Each row in the dataframe represents the attributions for an instance, and
            each column corresponds to a feature.
        instance (int): Index of the specific instance for which to compute the agreements.
        methods (list of str): A list of names of the explainability methods (e.g., ['LIME', 'SHAP', 'IG']).

    Returns:
        tuple of np.ndarray: Two square matrices (n_methods x n_methods):
            - `feature_agree`: Matrix where (i, j) represents the feature agreement
               between method `i` and method `j`.
            - `rank_agree`: Matrix where (i, j) represents the rank agreement
               between method `i` and method `j`.

    Example:
        weights_dfs = [df_lime, df_shap, df_ig]
        methods = ['LIME', 'SHAP', 'IG']
        instance_idx = 2

        feature_agreement_matrix, rank_agreement_matrix = compute_matrices(
            weights=weights_dfs,
            instance=instance_idx,
            methods=methods
        )

        print("Feature Agreement Matrix:")
        print(feature_agreement_matrix)

        print("Rank Agreement Matrix:")
        print(rank_agreement_matrix)
    """
    n_rankings = len(methods)

    feature_agree = np.zeros((n_rankings, n_rankings))
    rank_agree = np.zeros((n_rankings, n_rankings))

    for i, j in itertools.product(range(n_rankings), range(n_rankings)):
        r1 = to_rankings(weights[i], instance)[:n]
        r2 = to_rankings(weights[j], instance)[:n]
        feature_agree[i, j] = feature_agreement(r1, r2)
        rank_agree[i, j] = rank_agreement(r1, r2)

    return feature_agree, rank_agree


def compute_average_matrices(weights, instances, methods, n=10):
    """
    Compute the average feature agreement and rank agreement matrices across multiple instances.

    Parameters:
    ----------------
    weights : list of DataFrame
        List of feature attribution DataFrames for each method.
    instances : list of int
        List of instance indices to evaluate.
    methods : list of str
        List of explanation methods being compared.

    Returns:
    ----------------
    avg_feature_agree : numpy.ndarray
        Average feature agreement matrix.
    avg_rank_agree : numpy.ndarray
        Average rank agreement matrix.
    """
    n_methods = len(methods)
    total_instances = len(instances)

    cumulative_feature_agree = np.zeros((n_methods, n_methods))
    cumulative_rank_agree = np.zeros((n_methods, n_methods))

    for instance in instances:
        feature_agree, rank_agree = compute_matrices(weights, instance, methods, n)
        cumulative_feature_agree += feature_agree
        cumulative_rank_agree += rank_agree

    avg_feature_agree = cumulative_feature_agree / total_instances
    avg_rank_agree = cumulative_rank_agree / total_instances
    return avg_feature_agree, avg_rank_agree


def compute_matrices_with_reference(weights, reference, instance, methods):
    """
    Compute feature and rank agreement matrices for multiple methods against a single reference.

    This function calculates the agreement between feature rankings obtained from
    various explainability methods and a reference method for a specific data instance.
    Two types of agreement are computed:
      - **Feature Agreement**: Measures the overlap between the top-k features of the
        rankings of each method compared to the reference.
      - **Rank Agreement**: Measures the order consistency of features between
        the rankings of each method and the reference.

    Parameters:
        weights (list of pd.DataFrame): A list of dataframes where each dataframe corresponds
            to feature attributions (weights) computed by a specific explainability method.
            Each row in the dataframe represents the attributions for an instance, and
            each column corresponds to a feature.
        reference (pd.DataFrame): A dataframe representing the attributions for the reference method.
            Each row in the dataframe represents the attributions for an instance,
            and each column corresponds to a feature.
        instance (int): Index of the specific instance for which to compute the agreements.
        methods (list of str): A list of names of the explainability methods (e.g., ['LIME', 'SHAP', 'IG']).

    Returns:
        tuple of np.ndarray: Two vectors of length `len(methods)`:
            - `feature_agree_with_ref`: Vector where the i-th element represents
              the feature agreement of method `i` compared to the reference.
            - `rank_agree_with_ref`: Vector where the i-th element represents
              the rank agreement of method `i` compared to the reference.

    Example:
        weights_dfs = [df_lime, df_shap, df_ig]
        reference_df = df_shap
        methods = ['LIME', 'SHAP', 'IG']
        instance_idx = 2

        feature_agree_with_reference, rank_agree_with_reference = compute_matrices_with_reference(
            weights=weights_dfs,
            reference=reference_df,
            instance=instance_idx,
            methods=methods
        )

        print("Feature Agreement with Reference:")
        print(feature_agree_with_reference)

        print("Rank Agreement with Reference:")
        print(rank_agree_with_reference)
    """
    n_rankings = len(methods)

    feature_agree_with_ref = np.zeros(n_rankings)
    rank_agree_with_ref = np.zeros(n_rankings)

    # Get rankings for the reference method
    ref_rankings = to_rankings(reference, instance)[:10]

    for i in range(n_rankings):
        # Get rankings for the current method
        r1 = ref_rankings
        r2 = to_rankings(weights[i], instance)[:10]

        # Compute agreements
        feature_agree_with_ref[i] = feature_agreement(r1, r2)
        rank_agree_with_ref[i] = rank_agreement(r1, r2)

    return feature_agree_with_ref, rank_agree_with_ref


def plot_feature_agreement(
    feature_agreement,
    methods: List[str],
    title: str,
    n: int = 10,
    output_path: str = None,
):
    """
    Allows you to plot the top features agreement between each methods and their attributions.

    Parameters
    ----------
    feature_agreement: avg_feature_agree or feature_agree computed with `compute_average_matrices` or `compute_matrices`
    methods: list of your methods that you want to compare
    title: plot title
    n: top n feature
    output_path: save plot to
    """
    corr = feature_agreement
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(
            corr,
            mask=mask,
            square=True,
            annot=True,
            annot_kws={"fontsize": 10},
            xticklabels=methods,
            yticklabels=methods,
            cmap="Reds",
            cbar=True,
        )
        ax.set_title(title, color="xkcd:medium blue", fontsize=18)
        ax.set_ylabel(
            f"Top Feature\nAgreement (N={n})", color="xkcd:medium blue", fontsize=18
        )
        ax.text(
            0.95,
            0.95,
            "(a)",
            fontsize=14,
            alpha=0.8,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        data = corr
        avg = np.mean(data[mask == 0])
        text = f"Avg. Agrmnt : {avg:.2f}"
        ax.annotate(
            text, (1.0, 0.84), xycoords="axes fraction", fontsize=14, ha="right"
        )
        avg = np.mean(data[4:, :4])
        text = f"Avg. Agrmnt b/t FI & FR: {avg:.2f}"
        ax.annotate(
            text, (1.0, 0.79), xycoords="axes fraction", fontsize=14, ha="right"
        )
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()


def plot_rank_agreement(rank_agreement, methods, title, n=10, output_path=None):
    """
    Allows you to plot the top rank agreement and their ranking between each methods and their attributions.

    Parameters
    ----------
    rank_agreement: avg_rank_agree or rank_agree computed with `compute_average_matrices` or `compute_matrices`
    methods: list of your methods that you want to compare
    title: plot title
    n: top n feature
    output_path: save plot to
    """
    corr = rank_agreement
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(
            corr,
            mask=mask,
            square=True,
            annot=True,
            annot_kws={"fontsize": 10},
            xticklabels=methods,
            yticklabels=methods,
            cmap="Blues",
        )
        ax.set_title(title, color="xkcd:medium blue", fontsize=18)
        ax.set_ylabel(
            f"Feature Rank\nAgreement (N={n})", color="xkcd:medium blue", fontsize=18
        )
        ax.text(
            0.95,
            0.95,
            "(b)",
            fontsize=14,
            alpha=0.8,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        data = corr
        avg = np.mean(data[mask == 0])
        text = f"Avg. Agrmnt : {avg:.2f}"
        ax.annotate(
            text, (1.0, 0.84), xycoords="axes fraction", fontsize=14, ha="right"
        )
        avg = np.mean(data[4:, :4])
        text = f"Avg. Agrmnt b/t FI & FR: {avg:.2f}"
        ax.annotate(
            text, (1.0, 0.79), xycoords="axes fraction", fontsize=14, ha="right"
        )
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()
