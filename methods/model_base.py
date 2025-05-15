import numpy as np
import torch

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP


class ModelBase:
    """Base class model for Malconv2 MLP."""

    def __init__(self, model: Extracted_MLP):
        self.model = model

    def predict(self, features):
        features = torch.tensor(features, dtype=torch.float32)
        return self.model(features).detach().numpy()

    def predict_probs(self, features):
        y_pred = self.predict(features)
        prob_class_1 = y_pred.squeeze()
        prob_class_0 = 1 - prob_class_1
        return np.column_stack((prob_class_0, prob_class_1))

    def predict_probs_with_dataframe(self, features):
        y_pred = self.predict(features.values)
        prob_class_1 = y_pred.squeeze()
        prob_class_0 = 1 - prob_class_1
        return np.column_stack((prob_class_0, prob_class_1))

    def predict_with_dataframe(self, features):
        y_pred = self.predict(features.values)
        prob_class_1 = y_pred.squeeze()
        return prob_class_1

    def predict_binary(self, features):
        y_pred = self.predict(features)
        prob_class_1 = y_pred.squeeze()
        return np.where(prob_class_1 >= 0.5, 1, 0)

    def to_tensor(self, features: np.array) -> torch.tensor:
        return torch.tensor(features, dtype=torch.float32).requires_grad_(True)

    def normalize_attributions(self, attributions) -> np.array:
        return attributions / np.max(np.abs(attributions))
