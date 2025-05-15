import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP
from methods.model_base import ModelBase


class LimeXAI(ModelBase):
    """LIME, Local Interpretable Model-agnostic Explanations, explains the predictions of any classifier in an interpretable and
    faithful manner, by learning an interpretable model locally around the prediction. Here, we use a TabularExplainer,
    for numerical features, perturb them by sampling from a Normal(0,1) and doing the inverse operation of mean-centering and
    scaling, according to the means and stds in the training data.
    Source: “Why Should I Trust You?” Explaining the Predictions of Any Classifier (https://arxiv.org/pdf/1602.04938)
    Code implementation: https://github.com/marcotcr/lime
    """

    def __init__(self, model: Extracted_MLP, X_train, y_train):
        """
        model: post convolutional model of Malconv2
        X_train: your train set to approximate your black box model.
        y_train: your train set labels
        """
        super().__init__(model)
        self.features_names = np.arange(0, 256)
        self.target_names = ["goodware", "malware"]
        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.features_names,
            class_names=self.target_names,
            feature_selection="highest_weights",  # allow to select most influent features in the model decision (regarding the num_features option)
            mode="classification",
            training_labels=y_train,
        )

    def attributions(self, features: np.array):
        """Returns LIME attributions for each input feature.
        features: post convolutional features
        """
        assert len(features.shape) > 1 and features.shape[1] >= 1, (
            "Features need to have 2D"
        )
        attributions = []
        for feat in tqdm(features):
            explanation = self.explainer.explain_instance(
                feat, self.predict_probs, num_features=len(self.features_names)
            )
            attr = list(explanation.as_map().values())
            max_index = max(item[0] for row in attr for item in row)
            scores = np.zeros(max_index + 1)
            for row in attr:
                for index, value in row:
                    scores[index] = value
            attributions.append(scores)
        return np.array(attributions)
