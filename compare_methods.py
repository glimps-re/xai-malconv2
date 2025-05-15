"""
compare_methods.py

Allows to compare XAI methods based on defined metrics: accuracy, stability, pgi, pgu, certainty, duration...
You should before generate the post-conv dataset based on the original train dataset.
(original_dataset -> convolutional model -> post-conv dataset -> mlp model -> explainability)
You could generate the post-conv dataset with `get_post_conv_dataset`.
"""

import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
import torch

from malconv.model import MalConv2Model
from methods.deep_shap import DeepSHAPXAI
from methods.deeplift import DeepLiftXAI
from methods.integrated_gradients import IntegratedGradientsXAI
from methods.kernel_shap import KernelSHAPXAI
from methods.lime import LimeXAI
from methods.surrogate import (
    LGBMExplainableModel,
    LinearExplainableModel,
    SGDExplainableModel,
    SurrogatesXAI,
)
from metrics.accuracy import compute_accuracy
from metrics.certainty import compute_certainty
from metrics.compactness import compute_compactness_with_threshold
from metrics.faithfulness import compute_pgi, compute_pgu
from metrics.rank_feature_agreement import (
    compute_average_matrices,
    plot_feature_agreement,
    plot_rank_agreement,
)
from metrics.stability import compute_determinism, compute_stability

# we specify here methods that are using additive principle on the black box model
_ADDITIVE_METHODS = [
    "integrated_gradients",
    "integrated_gradients_gw",
    "integrated_gradients_mw",
]


class WhichExplanationMakeSense:
    """
    WhichExplanationMakeSense allows to compare XAI methods based on differents metrics like accuracy, certainty,
    stability, determinism, duration, PGI, PGU... Here we used binary classification.
    Regarding your use case you can add more metrics or more methods.
    """

    def __init__(self, train_path, test_path, model_path, output_dir):
        self.post_conv_train_set = pd.read_parquet(train_path)
        self.post_conv_test_set = pd.read_parquet(test_path)
        self.model = MalConv2Model(model_name=model_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # define backgrounds randomly
        # background with 100 goodwares
        self.background_gw = np.array(
            self.post_conv_train_set[self.post_conv_train_set["target"] == 0]
            .sample(n=100, random_state=42)
            .values[:, 1:-1],
            dtype=np.float32,
        )
        # background with 100 malwares
        self.background_mlw = np.array(
            self.post_conv_train_set[self.post_conv_train_set["target"] == 1]
            .sample(n=100, random_state=42)
            .values[:, 1:-1],
            dtype=np.float32,
        )

        # define baselines based on backgrounds
        # goodware baseline
        self.baseline_gw = torch.tensor(
            np.mean(self.background_gw, axis=0), dtype=torch.float32
        ).reshape(1, -1)
        # malware baseline
        self.baseline_mlw = torch.tensor(
            np.mean(self.background_mlw, axis=0), dtype=torch.float32
        ).reshape(1, -1)

        # define datasets to trained surrogates models
        self.X_train = self.post_conv_train_set.iloc[:, 1:-1].values.astype(np.float32)
        self.y_train = self.post_conv_train_set.iloc[:, -1].values
        # define test set to evaluate methods attributions
        self.X_test = self.post_conv_test_set.iloc[:, 1:-1].values
        # create XAI methods
        self.instanciate_methods()

    def instanciate_methods(self):
        """
        Instanciate XAI methods with train set, backgrounds, baselines...
        """
        print("Instanciating methods...")
        self.methods = {
            "deeplift": DeepLiftXAI(model=self.model.mlp_model),
            "deeplift_gw": DeepLiftXAI(
                model=self.model.mlp_model, baseline=self.baseline_gw
            ),
            "deeplift_mw": DeepLiftXAI(
                model=self.model.mlp_model, baseline=self.baseline_mlw
            ),
            "integrated_gradients": IntegratedGradientsXAI(model=self.model.mlp_model),
            "integrated_gradients_gw": IntegratedGradientsXAI(
                model=self.model.mlp_model, baseline=self.baseline_gw
            ),
            "integrated_gradients_mw": IntegratedGradientsXAI(
                model=self.model.mlp_model, baseline=self.baseline_mlw
            ),
            "deepshap": DeepSHAPXAI(model=self.model.mlp_model),
            "deepshap_gw": DeepSHAPXAI(
                model=self.model.mlp_model, background=torch.tensor(self.background_gw)
            ),
            "deepshap_mw": DeepSHAPXAI(
                model=self.model.mlp_model, background=torch.tensor(self.background_mlw)
            ),
            "lime": LimeXAI(
                model=self.model.mlp_model, X_train=self.X_train, y_train=self.y_train
            ),
            # /!\ Kernel SHAP computation time can be very high, we used this method here to compare to the state of the art.
            "kernel_shap": KernelSHAPXAI(model=self.model.mlp_model),
            "kernel_shap_gw": KernelSHAPXAI(
                model=self.model.mlp_model, background=self.background_gw
            ),
            "kernel_shap_mw": KernelSHAPXAI(
                model=self.model.mlp_model, background=self.background_mlw
            ),
            "sgd_surrogate": SurrogatesXAI(
                model=self.model.mlp_model,
                surrogate_model=SGDExplainableModel,
                X_train=self.X_train,
            ),
            "lgbm_surrogate": SurrogatesXAI(
                model=self.model.mlp_model,
                surrogate_model=LGBMExplainableModel,
                X_train=self.X_train,
            ),
            "linear_surrogate": SurrogatesXAI(
                model=self.model.mlp_model,
                surrogate_model=LinearExplainableModel,
                X_train=self.X_train,
            ),
        }
        print("All XAI methods are ready to compute.")

    def evaluate_surrogates(self) -> dict:
        """Evaluate Surrogates methods with their teacher model. You should keep the most accurate models.
        Global surrogate models should approximate your black box model as close as possible.

        Returns
        accuracy_score: dict
        """
        if self.methods is not None:
            return {
                "lgbm_surrogate": self.methods["lgbm_surrogate"].evaluate(self.X_test),
                "sgd_surrogate": self.methods["sgd_surrogate"].evaluate(self.X_test),
                "linear_surrogate": self.methods["linear_surrogate"].evaluate(
                    self.X_test
                ),
            }
        else:
            raise ValueError("XAI methods should be instanciate before.")

    def compute_attributions(self):
        """
        Compute attributions for each XAI methods on your test dataset.
        """
        self._attr = {}
        for key, method in self.methods.items():
            print(f"Computing attributions for {key}...")
            self._attr[key] = method.attributions(self.X_test)

    def evaluate_with_metrics(self):
        """
        Apply each metric to XAI methods attributions and save results. Allows you to compare methods based on your criteria.
        """
        # get preds from model
        y_preds_from_model = (
            self.model.mlp_model(torch.tensor(self.X_test)).squeeze().detach().numpy()
        )
        # we keep for certainty only "strong" malware predictions
        certainty_indices = np.where(y_preds_from_model > 0.90)[0]

        self._metrics = {}
        for method, attributions in self._attr.items():
            self._metrics[method] = {}
            print(f"Computing accuracy for {method}...")
            if method in _ADDITIVE_METHODS:
                y_ref = 0
                if self.methods[method].baseline is not None:
                    y_ref = (
                        self.methods[method].model(self.methods[method].baseline).item()
                    )
                self._metrics[method]["accuracy"] = compute_accuracy(
                    attributions, y_preds_from_model, additivity=True, reference=y_ref
                )
            else:
                self._metrics[method]["accuracy"] = compute_accuracy(
                    attributions, y_preds_from_model
                )
            print(f"Computing certainty for {method}...")
            self._metrics[method]["certainty"] = np.mean(
                compute_certainty(attributions[certainty_indices])[0]
            )
            print(f"Computing stability for {method}...")
            stability = compute_stability(
                x=self.X_test,
                selection=np.arange(attributions.shape[0]),
                contributions=attributions,
                model=self.model.mlp_model,
                n_neighbors=5,
            )

            self._metrics[method]["stability"] = np.mean(
                stability["variability"], axis=0
            ).mean()
            print(f"Computing PGI for {method}...")
            self._metrics[method]["pgi"] = compute_pgi(
                X=self.X_test, model=self.model.mlp_model, attributions=attributions
            )[1]
            print(f"Computing PGU for {method}...")
            self._metrics[method]["pgu"] = compute_pgu(
                X=self.X_test, model=self.model.mlp_model, attributions=attributions
            )[1]
            print(f"Computing compacity for {method}...")
            self._metrics[method]["compacity"] = compute_compactness_with_threshold(
                attributions
            )[0].mean()
            # compute determinism and duration for each method, iterate 10 loop on X_test
            # n may be increase but computation time is exponential
            print(f"Computing determinism for {method}...")
            avg_determinism, avg_duration = compute_determinism(
                instances=self.X_test,
                attribution_fn=self.methods[method].attributions,
                n=10,
            )
            self._metrics[method]["determinism"] = avg_determinism
            self._metrics[method]["duration"] = avg_duration

            metrics_df = pd.DataFrame.from_dict(self._metrics, orient="index")
            metrics_df.to_parquet(f"{self.output_dir}/xai_metrics_dataframe.parquet")
            print(metrics_df.to_markdown())

        metrics_df = pd.DataFrame.from_dict(self._metrics, orient="index")
        metrics_df.to_parquet(f"{self.output_dir}/xai_metrics_dataframe.parquet")
        print("All metrics computed and saved successfully.")
        print(metrics_df.to_markdown())

    def save_agreement(self, n=50):
        """
        Save feature and rank agreement between XAI methods to output dir.
        n: int = top n features to consider
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        attributions = [
            pd.DataFrame(attr, columns=np.arange(256)) for attr in self._attr.values()
        ]
        print("Computing feature and ranking agreement between methods...")
        avg_feature_agree, avg_rank_agree = compute_average_matrices(
            attributions,
            instances=np.arange(self.X_test.shape[1]),
            methods=list(self._attr.keys()),
            n=n,
        )
        plot_feature_agreement(
            avg_feature_agree,
            methods=list(self._attr.keys()),
            title="Average feature agreement for malware / goodware",
            n=n,
            output_path=f"{self.output_dir}/avg_feature_agreement_{current_date}.png",
        )
        plot_rank_agreement(
            avg_rank_agree,
            methods=list(self._attr.keys()),
            title="Average rank agreement for malware / goodware",
            n=n,
            output_path=f"{self.output_dir}/avg_rank_agreement_{current_date}.png",
        )
        print(
            f"Feature and ranking agreement plots successfully saved to {self.output_dir}"
        )


@click.command()
@click.option(
    "-t",
    "--train-set-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the train post conv dataset (Parquet).",
)
@click.option(
    "-T",
    "--test-set-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the test post conv dataset (Parquet).",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained model file.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    default="output",
    help="Path to save the explainability results.",
)
def main(train_set_path, test_set_path, model_path, output_path):
    explanation = WhichExplanationMakeSense(
        train_path=train_set_path,
        test_path=test_set_path,
        model_path=model_path,
        output_dir=output_path,
    )
    explanation.compute_attributions()
    explanation.evaluate_with_metrics()
    explanation.save_agreement()


if __name__ == "__main__":
    main()
