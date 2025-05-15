import contextlib
from functools import partialmethod

import joblib
import numpy as np
import torch
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def convert_k_to_int(k, n_feat):
    """
    Return the range of k values to evaluate
    :param k: int, float, or str
        :setting k to -1 will evaluate all features
        :setting k to a float will evaluate the top k% of features (rounded up)
    :param n_feat: int, number of features
    :return: int
    """
    if k == -1:
        return n_feat
    if not isinstance(k, int):
        if isinstance(k, float):
            if 0 < k < 1:
                return np.ceil(k * n_feat).astype(int)
            else:
                raise ValueError(f"Float value for k {k} must be between 0 and 1")
        else:
            raise ValueError(f"Invalid type for k: {type(k)}")


def construct_param_string(params):
    """
    Construct a string from a dictionary of parameters
    :param params: dict
    :return: str
    """
    param_str = (
        "_" + "_".join([f"{k}_{v}" for k, v in params.items()]) if params else ""
    )
    return param_str


def convert_to_numpy(x):
    """Converts input to numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()  # in case of GPU
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def convert_to_tensor(x):
    """Converts input to torch tensor."""
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    return x


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def generate_mask(explanation, top_k):
    if not isinstance(explanation, torch.Tensor):
        explanation = torch.Tensor(explanation)
    mask_indices = torch.topk(explanation.abs(), top_k).indices
    mask = torch.ones(explanation.shape, dtype=bool)
    for i in mask_indices:
        mask[i] = False
    return mask
