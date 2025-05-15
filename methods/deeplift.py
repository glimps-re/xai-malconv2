import numpy as np
import torch
from captum.attr import DeepLift

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP
from methods.model_base import ModelBase


class DeepLiftXAI(ModelBase):
    """DeepLIFT explains the difference in output from some ‘reference’ output in terms of the difference of the input from
    some ‘reference’ input. The ‘reference’ input represents some default or ‘neutral’ input that is chosen according to
    what is appropriate for the problem at hand.
    Source: `Learning Important Features Through Propagating Activation Differences` (https://arxiv.org/pdf/1704.02685)
    Code implementation : https://captum.ai/docs/attribution_algorithms
    """

    def __init__(self, model: Extracted_MLP, baseline: torch.tensor = None):
        """
        model: post convolutional model of Malconv2
        baseline: your baseline to consider, it can be post convolutional features from goodwares or malwares regarding your objective.
            Default, baseline is null.
        """
        super().__init__(model)
        self.explainer = DeepLift(self.model)
        self.baseline = baseline

    def attributions(self, features: np.ndarray):
        """Returns DeepLift attributions for each input feature.
        features: post convolutional features
        """
        features = torch.tensor(features, dtype=torch.float32).requires_grad_(True)
        # null baseline by default if none
        attributions, _ = self.explainer.attribute(
            features, baselines=self.baseline, target=0, return_convergence_delta=True
        )
        return attributions.detach().numpy()
