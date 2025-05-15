import numpy as np
from interpret_community.mimic.mimic_explainer import MimicExplainer, ModelTask
from interpret_community.mimic.models import (
    BaseExplainableModel,
    SGDExplainableModel,
)
from sklearn.metrics import accuracy_score

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP
from methods.model_base import ModelBase


class SurrogatesXAI(ModelBase):
    """
    A global surrogate model is an interpretable model that is trained to approximate the predictions of a black box model. We can draw conclusions
    about the black box model by interpreting the surrogate model.
    The mimic explainer trains an explainable model to reproduce the output of the given black box model.
    The explainable model is called a surrogate model and the black box model is called a teacher model.
    Once trained to reproduce the output of the teacher model, the surrogate model's explanation can
    be used to explain the teacher model.

    Source: https://christophm.github.io/interpretable-ml-book/global.html
    Code implementation: https://github.com/interpretml/interpret-community
    """

    def __init__(
        self,
        model: Extracted_MLP,
        X_train: np.array,
        surrogate_model: BaseExplainableModel = SGDExplainableModel,
        is_function=False,
    ):
        """
        model: post convolutional model of Malconv2.
        X_train: your train set to approximate your black box model.
        surrogate_model: surrogate model to use to approximate your black box model, could be LinearExplainableModel, DecisionTreeExplainableModel,
            SGDExplainableModel, LGBMExplainableModel.
        """
        super().__init__(model)
        self.explainer = MimicExplainer(
            model,
            X_train,
            surrogate_model,
            augment_data=False,
            features=np.arange(0, 256),
            is_function=is_function,
            model_task=ModelTask.Classification,
            classes=["malware"],
        )

    def attributions(self, features: np.array):
        """Returns Surrogate model attributions for each input feature.
        features: post convolutional features
        """
        local_exp = self.explainer.explain_local(features)
        attributions = np.array(local_exp.local_importance_values[1])  # classe 1
        # attributions = attributions / np.max(np.abs(attributions)) # normalize
        return attributions

    def evaluate(self, evaluation_examples: np.array):
        """Returns accuracy score between your surrogate model and your teacher model (your original model that you want to explain).
        evaluation_examples: your test set
        """
        surrogate_model_preds = self.explainer._get_surrogate_model_predictions(
            evaluation_examples
        )
        teacher_preds = self.explainer._get_teacher_model_predictions(
            evaluation_examples
        )
        acc_score = accuracy_score(
            np.where(np.array(surrogate_model_preds) >= 0.5, 1, 0), teacher_preds
        )
        return acc_score
