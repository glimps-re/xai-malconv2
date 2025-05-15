from typing import Optional

import numpy as np
import torch
from captum.attr import IntegratedGradients

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP
from methods.model_base import ModelBase


class IntegratedGradientsXAI(ModelBase):
    """
    Integrated Gradients is an axiomatic model interpretability algorithm that assigns an importance score to each input feature by approximating
    the integral of gradients of the model’s output with respect to the inputs along the path (straight line) from given baselines / references to inputs.
    Source: Axiomatic Attribution for Deep Networks (https://arxiv.org/pdf/1703.01365)
    Code Implementation: https://captum.ai/docs/extension/integrated_gradients
    """

    def __init__(self, model: Extracted_MLP, baseline: Optional[torch.tensor] = None):
        """
        model: post convolutional model of Malconv2
        baseline: Baselines define the starting point from which integral is computed.
        """
        super().__init__(model)
        if baseline is None:
            baseline = torch.zeros(1, 256)
        self.explainer = IntegratedGradients(model)
        self.baseline = baseline

    def attributions(self, features: np.ndarray):
        """Returns Integrated Gradients attributions for each input feature.
        features: post convolutional features
        """
        features = torch.tensor(features, dtype=torch.float32).requires_grad_(True)
        attributions, _ = self.explainer.attribute(
            features, baselines=self.baseline, target=0, return_convergence_delta=True
        )
        return attributions.detach().numpy()
