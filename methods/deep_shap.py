from typing import Optional

import numpy as np
import shap
import torch

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP
from methods.model_base import ModelBase


class DeepSHAPXAI(ModelBase):
    """Meant to approximate SHAP values for deep learning models.
    This is an enhanced version of the DeepLIFT algorithm (Deep SHAP) where, similar to Kernel SHAP, we
    approximate the conditional expectations of SHAP values using a selection of background samples.
    Source: https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
    Code implementation : https://github.com/shap/shap/blob/master/shap/explainers/_deep/__init__.py
    """

    def __init__(self, model: Extracted_MLP, background: Optional[torch.tensor] = None):
        """
        model: post convolutional model of Malconv2
        background: the background dataset to use for integrating out features. You should only use something like 100 or 1000 random
            background samples, not the whole training dataset
        """
        super().__init__(model)
        if background is None:
            background = torch.zeros(1, 256)
        self.explainer = shap.DeepExplainer(model, background)

    def attributions(self, features: np.ndarray):
        """Returns DeepSHAP attributions for each input feature.
        features: post convolutional features
        """
        features = torch.tensor(features, dtype=torch.float32)
        attributions = self.explainer.shap_values(features, check_additivity=False)
        return attributions
