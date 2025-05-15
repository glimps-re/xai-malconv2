# Standard
from datetime import datetime as dt
from pathlib import Path

# Third party
import click
import pandas as pd
import torch
import tqdm
from binaryLoader import pad_collate_func
from MalConvGCT_nocat import MalConvGCT
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

output_dir = Path("your_output_path_to_save_model")
train_file = Path("your_train_set.parquet")
valid_file = Path("your_valid_set.parquet")
malconv_db = Path("path_to_saved_tensors")

# 10 epochs max
EPOCHS = 10


class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, directory: Path) -> None:
        self.dataframe = dataframe
        if not isinstance(directory, Path):
            directory = Path(directory)
        self.directory = directory
        self.file_names = dataframe["user_hash"].values

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        # we load a single file for each sample
        file_path = self.directory / f"{self.file_names[idx]}.pt"
        data = torch.load(file_path)
        x, y = data[0], data[1]
        return x.squeeze(), torch.tensor([y])


class MalConv2Training:
    def __init__(self, device: str, output_dir: Path, batch_size: int = 64) -> None:
        self.device = device
        self.batch_size = batch_size
        self.date = dt.now().strftime("%Y-%m-%d_%H-%M")
        model_desc = f"malconv2_batch_{batch_size}_{self.date}"
        self.writer = SummaryWriter(log_dir=output_dir / "runs" / model_desc)
        self.output_dir = output_dir / "models" / model_desc

        # Initialize the model and the learning process
        self.reset_network_state()

    def reset_network_state(self) -> None:
        """Reset the model and objects related to the learning process."""
        self.model = MalConvGCT(channels=256, window_size=256, stride=64)
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters())
        self.epoch = 0
        self.step = 0

    def _save_checkpoint(self) -> None:
        """Save actual model state to a dedicated torch file"""
        model_path = self.output_dir / f"epoch_{self.epoch}_step_{self.step}.checkpoint"
        if not model_path.parent.exists():
            model_path.parent.mkdir()
        torch.save(
            {
                "epoch": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            },
            model_path,
        )

    def _load_data(
        self,
        in_data_file: Path,
        out_data_file: Path = malconv_db,
        shuffle: bool = True,
        batch_size: int = None,
    ) -> DataLoader:
        data = pd.read_parquet(in_data_file)
        data_ds = CustomDataset(data, out_data_file)
        if not batch_size:
            batch_size = self.batch_size
        return DataLoader(
            data_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_func
        )

    def compute_rates_on_valid(self, valid_dataloader: DataLoader) -> dict[str, float]:
        """Return FP/FN rate on the evaluation dataset"""
        # Put model in evaluation mode
        self.model.eval()

        nb_correct, nb_traited = 0, 0
        with torch.no_grad():
            for batch in valid_dataloader:
                samples, targets = batch
                samples, targets = samples.to(self.device), targets.to(self.device)
                targets = targets.to(torch.int64)
                pred, _, _ = self.model(samples)
                pred = pred.detach().cpu()
                targets = targets.detach().cpu()

                _, predicted = torch.max(pred.data, 1)
                nb_correct += (predicted == targets).sum().item()
                nb_traited += targets.size(0)

        avg_valid_acc = 1.0 * nb_correct / nb_traited

        # Put back model in training mode
        self.model.train()

        return {"avg_valid_acc": avg_valid_acc}

    def _train_one_epoch(
        self, train_dataloader: DataLoader, valid_dataloader: DataLoader, max_step: int
    ) -> None:
        """Train the Neural Network for one epoch."""
        # Instanciate processing bar
        self.epoch += 1
        avg_train_acc, avg_valid_acc = 0.0, 0.0
        nb_correct, nb_traited = 0, 0
        batch_bar = tqdm.tqdm(
            train_dataloader,
            leave=False,
            postfix=f"avg_train_acc={avg_train_acc:.2f} avg_valid_acc={avg_valid_acc:.2f}",
            smoothing=0.01,
        )
        # Put the model in training mode
        self.model.train()
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.output_dir / "runs/"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for batch in batch_bar:
                prof.step()
                self.opt.zero_grad()
                samples, targets = batch
                samples = samples.to(self.device)
                targets = targets.to(self.device).to(torch.long)

                # perform a forward pass and calculate the training loss
                pred, _, _ = self.model(samples)
                loss = torch.nn.functional.cross_entropy(
                    pred, targets, reduction="mean"
                )
                loss.backward()
                self.opt.step()

                pred = pred.detach().cpu()
                targets = targets.detach().cpu()
                _, predicted = torch.max(pred.data, 1)
                nb_correct += (predicted == targets).sum().item()
                nb_traited += targets.size(0)

                avg_train_acc = 1.0 * nb_correct / nb_traited
                self.writer.add_scalar("acc/train", avg_train_acc, self.step)

                # Return if we excedeed the maximal number of step
                self.step += 1
                if self.step >= max_step:
                    return

                # We compute results in validation set only each 100 steps
                # as the process is time consuming.
                if self.step % 1000 == 0:
                    self._save_checkpoint()

                # /!\ we skip the validation phase to speed up the training process

                #    # Compute evaluation
                #    results = self.compute_rates_on_valid(valid_dataloader)
                #    avg_valid_acc = results["avg_valid_acc"]

                #    self.writer.add_scalar("acc/valid", avg_valid_acc, self.step)

                # Update processing bar results
                batch_bar.set_postfix(
                    avg_train_acc=avg_train_acc, avg_valid_acc=avg_valid_acc
                )

    def train_network(
        self, train_file: Path, valid_file: Path, max_epoch=EPOCHS, max_step=51000
    ):
        """Train a combined Neural Network on given data for at most `max_epoch` epochs or `max_step` steps."""
        # Ensure that we start a learning process from scratch
        if self.step > 0:
            self.reset_network_state()

        # Load data
        train_dataloader = self._load_data(train_file, shuffle=True)
        valid_dataloader = self._load_data(valid_file, shuffle=False)

        # Train the neural network with warm restart of the learning rate
        for _ in range(0, max_epoch):
            self._train_one_epoch(train_dataloader, valid_dataloader, max_step)

            if self.step >= max_step or self.epoch >= max_epoch:
                print("Maximal number of step reached --> stop learning process.")
                return

        self._save_checkpoint()
        return self.model


@click.command()
@click.option(
    "-b",
    "--batch-size",
    default=128,
    required=False,
    help="Batch size to use during training process",
    show_default=True,
)
def main(batch_size: int = 128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(15)
    training = MalConv2Training(device, output_dir, batch_size=batch_size)
    training.train_network(train_file, valid_file, max_epoch=EPOCHS)
    print("Model trained successfully!")


if __name__ == "__main__":
    main()
