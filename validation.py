import os
from pathlib import Path

import numpy as np
import pandas as pd
import click
import torch
from MalConvGCT_nocat_Inf import MalConvGCT
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm as tqdmnb


class CustomDataset(Dataset):
    def __init__(self, dataframe, directory):
        self.dataframe = dataframe
        self.directory = directory
        self.file_names = [
            f
            for f in os.listdir(directory)
            if f.endswith(".pt") and f[:-3] in dataframe["user_hash"].values
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # we load a single file for each sample
        file_path = os.path.join(self.directory, self.file_names[idx])
        data = torch.load(file_path)
        x, y = data[0], data[1]
        return x.squeeze(), torch.tensor([y])


def pad_collate_func(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to
    pad out files in a batch to the length of the longest item in the batch.
    """
    vecs = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
    # stack will give us (B, 1), so index [:,0] to get to just (B)
    y = torch.stack(labels)[:, 0]

    return x, y


class Malconv2Validation:
    """Allows you to evaluate your malconv trained model on your validation set."""

    def __init__(
        self,
        model_path: Path,
        ds_valid_path: Path,
        malconv_db_path: Path,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.malconv_db_path = malconv_db_path
        self.mlgct = MalConvGCT(
            channels=256,
            window_size=256,
            stride=64,
        )
        _model_dict = torch.load(model_path)
        self.mlgct.load_state_dict(_model_dict["model_state_dict"], strict=False)
        self.mlgct.eval()
        self.mlgct.to(device)
        self.ds_valid = pd.read_parquet(ds_valid_path)

    def evaluate(self):
        """Run evaluation on your valid dataset."""
        stats = dict()
        for format_ in ["PE", "ELF"]:
            print(format_)
            ds_selection = self.ds_valid[(self.ds_valid["type"] == format_)]
            malware_ratio = (
                len(ds_selection[ds_selection["target"] == 1])
                / len(ds_selection)
                * 100.0
            )
            dataloader_ = DataLoader(
                CustomDataset(ds_selection, self.malconv_db_path),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=pad_collate_func,
            )

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            tot_pos = 0
            tot_neg = 0
            total = 0
            correct = 0

            true_labels = []
            predictions = []

            for x, y in tqdmnb(dataloader_, leave=False):
                x.to(self.device)
                pred, _, _ = self.mlgct(x)
                pred = pred.detach().cpu()
                _, predicted = torch.max(pred.data, 1)
                total += y.size(0)
                tot_pos += (y == 1).sum().item()
                tot_neg += (y == 0).sum().item()
                correct += (predicted == y).sum().item()
                tp += ((y == 1) & (predicted == y)).sum().item()
                tn += ((y == 0) & (predicted == y)).sum().item()
                fp += ((predicted == 1) & (predicted != y)).sum().item()
                fn += ((predicted == 0) & (predicted != y)).sum().item()
                true_labels.extend(y.tolist())
                predictions.extend(
                    torch.softmax(pred, 1)[:, 1].tolist()
                )  # get probability for class 1

            auc_score = roc_auc_score(true_labels, predictions)
            print(f"AUC Score: {auc_score:.4f}")
            # compute the TP @ 0.5% FP
            coupl = list(zip(true_labels, predictions))
            coupl = sorted(coupl, key=lambda k: k[1], reverse=True)
            coupl = np.array(coupl)

            threshold = coupl[coupl[:, 0] == 0][
                int(len(coupl[coupl[:, 0] == 0]) * 0.005), 1
            ]
            fp_ = (
                len(coupl[(coupl[:, 0] == 0) & (coupl[:, 1] > threshold)])
                / len(coupl[coupl[:, 0] == 0])
                * 100
            )
            tp_ = (
                len(coupl[(coupl[:, 0] == 1) & (coupl[:, 1] > threshold)])
                / len(coupl[coupl[:, 0] == 1])
                * 100
            )
            print(
                f"with threshold = {threshold} there is {fp_:.3}% FP and {tp_:.3}% TP"
            )
            print(
                "\taccuracy",
                1.0 * correct / total,
                "from",
                total,
                f"binaries with {malware_ratio:.3}% malwares",
            )
            print("\ttp", 1.0 * tp / tot_pos)
            print("\ttn", 1.0 * tn / tot_neg)
            print("\tfp", 1.0 * fp / tot_neg)
            print("\tfn", 1.0 * fn / tot_pos)
            stats[format_] = [
                total,
                auc_score,
                correct / total,
                fp / tot_neg,
                fn / tot_pos,
            ]
            return stats


@click.command()
@click.option("--model-path", required=True, help="Path to the model file.")
@click.option("--ds-valid-path", required=True, help="Path to the validation dataset.")
@click.option(
    "--malconv-db-path", required=True, help="Path to the Malconv torch files database."
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    required=True,
    help="Device to use for evaluation (cpu or cuda).",
)
@click.option(
    "--batch-size", type=int, required=True, help="Batch size for evaluation."
)
def main(model_path, ds_valid_path, malconv_db_path, device, batch_size):
    """Evaluate the Malconv2 model using the provided parameters."""
    validation = Malconv2Validation(
        model_path, ds_valid_path, malconv_db_path, device, batch_size
    )
    validation.evaluate()


if __name__ == "__main__":
    main()
