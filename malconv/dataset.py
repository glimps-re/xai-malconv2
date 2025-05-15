import logging as logger
import os
from pathlib import Path

import lief
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from malconv.model import MalConv2Model, get_exec_sections_elf, get_exec_sections_pe


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


def pad_collate_func(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to pad out files in a batch to the length of the longest item in the batch.
    """
    vecs = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
    # stack will give us (B, 1), so index [:,0] to get to just (B)
    y = torch.stack(labels)[:, 0]

    return x, y


def get_post_conv_dataset(original_set_path: str, model_path: str, torch_db_path: str) -> pd.DataFrame:
    """
    Allows to create a post-conv dataset. This dataset is used by explainability methods and mlp model.
    Parameters
    ----------
    original_set_path: str = path to your original dataset (with user_hash, target)
    model_path: str = path to your malconv model
    torch_db_path: str = path to your torch database which contains .pt files with executable code of each binary

    Returns
    -------
    post_conv_dataset: pd.DataFrame
    """
    original_dataset = pd.read_parquet(original_set_path)
    model = MalConv2Model(model_name=model_path)
    dataset = CustomDataset(original_dataset, Path(torch_db_path))
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_func)
    post_conv_dataset = pd.DataFrame(
        [
            [dataset.file_names[idx], *model.model(batch[0])[0].squeeze().detach().numpy(), batch[1].item()]
            for idx, batch in tqdm(enumerate(train_dataloader))
        ],
        columns=["user_hash", *np.arange(256), "target"],
    )
    return post_conv_dataset


class Malconv2RawBytes:
    """This class allows you to create a training or validation set.
    It extracts executable raw bytes from your binaries."""

    def __init__(self, sha256_list: list[str], targets: list[int], input_dir: str, output_path: str) -> None:
        """
        sha256_list: list of your binaries to compute
        targets: list of your labels 0 for goodware 1 for malware
        input_dir: input directory where are stored your binaries
        output_path: output dir where to store your torch files with executable code of each binary.
        """
        self.sha256_list = sha256_list
        self.targets = targets
        self.output_path = output_path
        self.input_dir = input_dir

    def process_file(self, filepath: str):
        """Process binary and extract raw bytes."""
        raw_bytes = []
        try:
            if lief.is_pe(filepath):
                raw_bytes, _ = get_exec_sections_pe(filepath)
            elif lief.is_elf(filepath):
                raw_bytes, _ = get_exec_sections_elf(filepath)
            else:
                logger.warning("WARN: your file is not recognized as an ELF or PE.")
                raw_bytes, _ = None, None
        except Exception as error:
            logger.error(error)
            raw_bytes, _ = None, None
        if raw_bytes is None:
            # random bytes if None
            x_np = np.random.randint(0, 256, size=(1, 2000), dtype=np.int16)
        else:
            x_np = np.frombuffer(bytearray(raw_bytes), dtype=np.uint8).astype(np.int16) + 1
        x_np = x_np.reshape((1, -1))
        if x_np.shape[1] < 2000:
            # pad to have a min. length equal to 2000
            x_np = np.pad(x_np, ((0, 0), (0, 2000 - x_np.shape[1])), "constant")
        processed_x = torch.Tensor(x_np).type(torch.int16)
        return processed_x

    def run(self):
        """Start raw bytes extraction on your binaries."""
        if os.path.exists(self.input_dir):
            for idx, binary in enumerate(self.sha256_list):
                processed_x = self.process_file(f"{self.input_dir}/{binary}")
                target = self.targets[idx]
                torch.save((processed_x, target), Path(self.output_path) / f"{binary}.pt")
            logger.info("Raw bytes extracted successfully from your binaries! Saved to %s", self.output_path)
        else:
            raise ValueError("Please, provide an existing path.")
