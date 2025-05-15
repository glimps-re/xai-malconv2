from typing import Optional

import numpy as np
import shap

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP
from methods.model_base import ModelBase


class KernelSHAPXAI(ModelBase):
    """Uses the Kernel SHAP method to explain the output of any function. Kernel SHAP is a method that uses a special weighted linear regression
    to compute the importance of each feature. The computed importance values are Shapley values from game theory and also coefficients from a local linear regression.
    Source : https://christophm.github.io/interpretable-ml-book/shapley.html
    Code implementation: https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
    """

    def __init__(self, model: Extracted_MLP, background: Optional[np.ndarray] = None):
        """
        model: post convolutional model of Malconv2
        background: the background dataset to use for integrating out features. To determine the impact of a feature, that feature is set
            to “missing” and the change in the model output is observed. Since most models aren’t designed to handle arbitrary missing data
            at test time, we simulate “missing” by replacing the feature with the values it takes in the background dataset.
        """
        super().__init__(model)
        if background is None:
            background = np.zeros(256).reshape(1, -1)
        self.explainer = shap.KernelExplainer(
            model=self.predict_probs, data=background, feature_names=np.arange(0, 256)
        )

    def attributions(self, features: np.ndarray):
        """Returns KernelSHAP attributions for each input feature.
        features: post convolutional features
        """
        assert len(features.shape) > 1 and features.shape[1] >= 1, (
            "Features need to have 2D"
        )
        attributions = np.array(self.explainer.shap_values(features))[
            1, :, :
        ]  # class 1
        return attributions
