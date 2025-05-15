"""
Faithfulness

Based on Chirag Agarwal et al. “Openxai: Towards a transparent evaluation of post hoc model explanations”,
we use two metrics (PGI and PGU) to measure the predictive faithfulness of the selected XAI methods.

PGI. For Prediction Gap on Important feature perturbation. It calculates the difference in prediction
probability that results from perturbing features considered as important by a given explanation. This
gives a measurement of how much the model’s prediction changes after important changes. We want
to maximize this value.

PGU. For Prediction Gap on Unimportant feature perturbation. This is the inverse of PGI, we perturb
the least important features given by the explainability method, and aim to minimize this value.

Note: we perturb here 25% (k=0.25) of the top of most (or less) important features, you can change it if necessary.
"""

import warnings

import numpy as np
import torch
from sklearn.metrics import auc

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.notebook import trange

import metrics.tools.experiment_utils as utils
from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP


def compute_faithfulness(
    explanations,
    inputs,
    model,
    k,
    perturb_method,
    feature_metadata,
    n_samples=100,
    invert=False,
    seed=-1,
    n_jobs=None,
    AUC=False,
):
    """
    Evaluates the faithfulness of feature attributions by measuring the change in model predictions
    when perturbing the top-k most (or least) important features according to the explanation.

    The function perturbs input features based on the provided perturbation method and computes the
    expected absolute difference in model predictions. If `AUC` is enabled, it integrates the metric
    over different values of k.

    :param explanations: torch.Tensor of shape (n_inputs, n_features)
        Feature attribution scores for each input sample.
    :param inputs: torch.Tensor of shape (n_inputs, n_features)
        Original input samples before perturbation.
    :param model: torch.nn.Module
        The model used for prediction.
    :param k: int
        Number of top-k features to be perturbed. If `invert=True`, the bottom-k features are perturbed instead.
    :param perturb_method: BasePerturbation
        An instance of a perturbation method class that generates perturbed inputs.
    :param feature_metadata: list[str]
        A list of feature types, where 'c' represents continuous and 'd' represents discrete features.
    :param n_samples: int, optional (default=100)
        Number of perturbed samples to generate per input.
    :param invert: bool, optional (default=False)
        Whether to invert the selection of top-k features. If True, perturbs the least important features (PGU metric).
    :param seed: int, optional (default=-1)
        Random seed for reproducibility. If -1, the seed is set to the instance index.
    :param n_jobs: int, optional (default=None)
        Number of parallel jobs for computation. If -1, uses all available CPU cores. If None, runs sequentially.
    :param AUC: bool, optional (default=False)
        Whether to compute the Area Under the Curve (AUC) over different values of k.

    :return: tuple (np.ndarray, float)
        - `metric_distr`: An array of faithfulness scores for each input.
        - `mean_metric`: The average faithfulness score across all inputs.

    :raises ValueError:
        If the shapes of `explanations` and `inputs` do not match.
    """
    # Preprocess
    explanations, inputs = (
        utils.convert_to_tensor(explanations),
        utils.convert_to_tensor(inputs),
    )
    if inputs.shape != explanations.shape:
        raise ValueError("The input and explanation shapes do not match.")
    n_inputs, n_features = explanations.shape
    max_k = utils.convert_k_to_int(k, n_features)

    # Compute metric
    params = [
        inputs,
        explanations,
        invert,
        n_jobs,
        [model, perturb_method, feature_metadata, n_samples, seed],
    ]
    if AUC and max_k > 1:
        metric_distr_all_ks = np.array(
            [
                _single_k_pred_faith(k, *params)
                for k in trange(
                    1, max_k + 1, desc="Calculating PGI/PGU across top-k values"
                )
            ]
        )
        metric_distr = np.array(
            [
                auc(np.arange(max_k) / (max_k - 1), metric_distr_all_ks[:, i])
                for i in range(n_inputs)
            ]
        )
    else:
        metric_distr = _single_k_pred_faith(max_k, *params)

    return metric_distr, np.mean(metric_distr)


metrics_dict = {"PGU": compute_faithfulness, "PGI": compute_faithfulness}

metrics_params = {"PGU": {"invert": True}, "PGI": {"invert": False}}


class BasePerturbation:
    """
    Base Class for perturbation methods.
    """

    def __init__(self, data_format):
        """
        Initialize generic parameters for the perturbation method
        """
        self.data_format = data_format

    def get_perturbed_inputs(self):
        """
        This function implements the logic of the perturbation methods which will return perturbed samples.
        """
        pass


class NormalPerturbation(BasePerturbation):
    def __init__(
        self,
        data_format,
        mean: int = 0,
        std_dev: float = 0.05,
        flip_percentage: float = 0.3,
    ):
        self.mean = mean
        self.std_dev = std_dev
        self.flip_percentage = flip_percentage

        super(NormalPerturbation, self).__init__(data_format)
        """
        Initializes the marginal perturbation method where each column is sampled from marginal distributions given per variable.
        dist_per_feature : vector of distribution generators (tdist under torch.distributions).
        Note : These distributions are assumed to have zero mean since they get added to the original sample.
        """
        pass

    def get_perturbed_inputs(
        self,
        original_sample: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
        num_samples: int,
        feature_metadata: list,
    ) -> torch.tensor:
        """
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        """
        feature_type = feature_metadata
        assert len(feature_mask) == len(original_sample), (
            f"mask size == original sample in get_perturbed_inputs for {self.__class__}"
        )

        continuous_features = torch.tensor([i == "c" for i in feature_type])
        discrete_features = torch.tensor([i == "d" for i in feature_type])

        # Processing continuous columns
        perturbations = (
            torch.normal(self.mean, self.std_dev, [num_samples, len(feature_type)])
            * continuous_features
            + original_sample
        )

        # Processing discrete columns
        flip_percentage = self.flip_percentage
        p = torch.empty(num_samples, len(feature_type)).fill_(flip_percentage)
        perturbations = perturbations * (~discrete_features) + torch.abs(
            (perturbations * discrete_features)
            - (torch.bernoulli(p) * discrete_features)
        )

        # keeping features static that are in top-K based on feature mask
        perturbed_samples = original_sample * feature_mask + perturbations * (
            ~feature_mask
        )

        return perturbed_samples


class FaithfulnessEvaluator:
    """
    Evaluates the faithfulness of feature attributions by computing a given metric
    for a specified model.

    This class provides an interface to compute faithfulness metrics, such as PGU and PGI,
    by applying the selected evaluation method to the model's predictions and explanations.

    :param model: torch.nn.Module
        The model to be evaluated.
    :param metric: str
        The name of the faithfulness metric to use. Must be a key in `metrics_dict`.

    :raises NotImplementedError:
        If the specified metric is not found in `metrics_dict`.

    Attributes:
    -----------
    - `model`: The model instance being evaluated.
    - `metric`: The selected faithfulness metric.
    - `metric_fn`: The function corresponding to the selected metric.
    - `metrics_params`: A dictionary of parameters associated with the metric.

    Methods:
    --------
    - `evaluate(**param_dict)`: Computes the selected faithfulness metric with the given parameters.
    """

    def __init__(self, model, metric):
        # set model and metric
        if metric not in metrics_dict:
            raise NotImplementedError(f"The metric {metric} is not implemented.")
        self.model = model
        self.metric = metric
        # faithfulness callback
        self.metric_fn = metrics_dict[metric]
        self.metrics_params = metrics_params[metric]

        if metric in metrics_dict:
            self.metrics_params["model"] = self.model

    def evaluate(self, **param_dict):
        """
        Computes the faithfulness metric using the given parameters.

        :param param_dict: dict
            Keyword arguments corresponding to the parameters required by the selected metric.

        :return: The computed metric value(s), as defined by the metric function.
        """
        self.metrics_params.update(param_dict)  # update metric_params with args
        return self.metric_fn(**self.metrics_params)


def _get_perturbation_explanations(
    model,
    input,
    explainer,
    perturb_method,
    n_samples,
    feature_metadata,
    n_perturbations,
):
    y_pred = torch.argmax(model(input.float()), dim=1)[0]
    explanation = explainer.get_explanations(input.float(), label=y_pred)
    # Get perturbations of the input that have the same prediction
    x_prime_samples = perturb_method.get_perturbed_inputs(
        original_sample=input[0].float(),
        feature_mask=torch.zeros(input.shape[-1], dtype=bool),
        num_samples=n_samples,
        feature_metadata=feature_metadata,
    )
    y_prime_preds = torch.argmax(model(x_prime_samples.float()), dim=1)
    ind_same_class = (y_prime_preds == y_pred).nonzero()[:n_perturbations].squeeze()
    x_prime_samples = torch.index_select(
        input=x_prime_samples, dim=0, index=ind_same_class
    )
    y_prime_samples = torch.index_select(
        input=y_prime_preds, dim=0, index=ind_same_class
    )

    # For each perturbation, calculate the explanation
    exp_prime_samples = utils.convert_to_numpy(
        explainer.get_explanations(x_prime_samples, label=y_prime_samples)
    )
    return x_prime_samples, exp_prime_samples, explanation


# eval_pred_faithfulness
def _single_k_pred_faith(k, inputs, explanations, invert, n_jobs, params):
    if n_jobs is not None:
        with utils.tqdm_joblib(
            tqdm(desc=f"Computing {'PGU' if invert else 'PGI'}", total=len(inputs))
        ) as _:
            metric_distr = Parallel(n_jobs=n_jobs)(
                delayed(_single_idx_pred_faith)(
                    i, input, explanation, k, invert, *params
                )
                for i, (input, explanation) in enumerate(zip(inputs, explanations))
            )
    else:
        metric_distr = np.array(
            [
                _single_idx_pred_faith(i, input, explanation, k, invert, *params)
                for i, (input, explanation) in enumerate(zip(inputs, explanations))
            ]
        )
    return metric_distr


# eval_pred_faithfulness
def _single_idx_pred_faith(
    i,
    input,
    explanation,
    k,
    invert,
    model,
    perturb_method,
    feature_metadata,
    n_samples,
    seed,
):
    """
    Inner loop computation extracted from eval_pred_faithfulness to enable parallelization.
    """
    top_k_mask = utils.generate_mask(explanation, k)
    top_k_mask = torch.logical_not(top_k_mask) if invert else top_k_mask

    # get perturbations of instance x
    torch.manual_seed(i if seed == -1 else seed)
    np.random.seed(i if seed == -1 else seed)
    x_perturbed = perturb_method.get_perturbed_inputs(
        original_sample=input,
        feature_mask=top_k_mask,
        num_samples=n_samples,
        feature_metadata=feature_metadata,
    )
    # Average the expected absolute difference.
    y = utils.convert_to_numpy(model(input.reshape(1, -1).float()))
    y_perturbed = utils.convert_to_numpy(model(x_perturbed.float()))
    return np.mean(np.abs(y_perturbed - y)[:, 0])


def compute_pgi(X: np.array, attributions: np.array, model: Extracted_MLP):
    """
    Computes the PGI (Prediction Gap on Important feature perturbation) metric to evaluate the faithfulness
    of feature attributions for a given model and specific XAI method.

    It calculates the difference in prediction probability that results from perturbing features considered
    as important by a given explanation. This gives a measurement of how much the model’s prediction changes
    after important changes. We want to maximize this value.

    :param X: np.array
        Input data of shape (n_samples, n_features) representing the original instances.
    :param attributions: np.array
        Attribution scores of shape (n_samples, n_features) indicating feature importance.
    :param model: Extracted_MLP
        The extracted MLP model whose attributions are being evaluated.

    :return: tuple (np.array, float)
        - The PGI scores for each instance. (between 0 and 1)
        - The mean PGI score across all instances.
    """

    metric = "PGI"
    param_dict = {
        "k": 0.25,  # Top-k percentage of features to consider
        "AUC": True,  # Whether to compute AUC of metric distribution
        "std": 0.1,  # Standard deviation for perturbation
        "n_samples": 100,  # Number of perturbed samples
        "seed": -1,  # Random seed
        "n_jobs": 1,  # Number of parallel jobs
    }
    param_dict["inputs"] = X
    param_dict["explanations"] = attributions

    # Compute flip percentage for normal perturbation
    flip = np.sqrt(2 / np.pi) * param_dict["std"]
    method = NormalPerturbation(
        "tabular", mean=0.0, std_dev=param_dict["std"], flip_percentage=flip
    )

    param_dict["perturb_method"] = method
    param_dict["feature_metadata"] = [
        "c"
    ] * 256  # 256 features (for MalConv post-convolution output) and C for continious

    del param_dict["std"]  # Remove std as it has been used to set up perturbation

    # Initialize evaluator and compute PGI
    evaluator = FaithfulnessEvaluator(model, metric)
    score, mean_score = evaluator.evaluate(**param_dict)

    return score, mean_score


def compute_pgu(X: np.array, attributions: np.array, model: Extracted_MLP):
    """
    Computes the PGU (Prediction Gap on Unimportant feature perturbation) metric to evaluate the faithfulness
    of feature attributions for a given model and specific XAI method.

    This is the inverse of PGI, we perturb the least important features given by the explainability method,
    and aim to minimize this value.

    :param X: np.array
        Input data of shape (n_samples, n_features) representing the original instances.
    :param attributions: np.array
        Attribution scores of shape (n_samples, n_features) indicating feature importance.
    :param model: Extracted_MLP
        The extracted MLP model whose attributions are being evaluated.

    :return: tuple (np.array, float)
        - The PGU scores for each instance. (between 0 and 1)
        - The mean PGU score across all instances.
    """

    metric = "PGU"
    param_dict = {
        "k": 0.25,  # Bottom-k percentage of features to consider
        "AUC": True,  # Whether to compute AUC of metric distribution
        "std": 0.1,  # Standard deviation for perturbation
        "n_samples": 100,  # Number of perturbed samples
        "seed": -1,  # Random seed
        "n_jobs": 1,  # Number of parallel jobs
    }
    param_dict["inputs"] = X
    param_dict["explanations"] = attributions

    # Compute flip percentage for normal perturbation
    flip = np.sqrt(2 / np.pi) * param_dict["std"]
    method = NormalPerturbation(
        "tabular", mean=0.0, std_dev=param_dict["std"], flip_percentage=flip
    )

    param_dict["perturb_method"] = method
    param_dict["feature_metadata"] = [
        "c"
    ] * 256  # 256 features (for MalConv post-convolution output) and C for continious

    del param_dict["std"]  # Remove std as it has been used to set up perturbation

    # Initialize evaluator and compute PGU
    evaluator = FaithfulnessEvaluator(model, metric)
    score, mean_score = evaluator.evaluate(**param_dict)

    return score, mean_score
