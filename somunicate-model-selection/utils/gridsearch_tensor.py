import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from utils.gridsearch import _get_fold_params, _validate_total_samples


def _compute_tensor_bin_edges(
    target_values: torch.Tensor, n_bins: int, non_parametric: bool
) -> torch.Tensor:
    """Computes bin edges for tensor stratification."""
    if non_parametric:
        edges_linear = np.linspace(0, 1, n_bins + 1)
        edges_np = np.quantile(target_values.numpy(), edges_linear)
        edges = torch.tensor(edges_np, dtype=torch.float32)
        edges[0] -= 1e-3
        edges[-1] += 1e-3
    else:
        edges = torch.linspace(
            target_values.min().item() - 1e-3,
            target_values.max().item() + 1e-3,
            n_bins + 1,
            dtype=torch.float32,
        )
    return edges


def _stratify_tensor_labels(
    target_tensor: torch.Tensor, n_bins: int, non_parametric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratifies tensor labels for train/test split."""
    # ensure two-dimensional tensor
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.unsqueeze(1)
    if target_tensor.shape[1] > 1:
        arr = PCA(n_components=1).fit_transform(target_tensor.numpy())
        target_array = torch.tensor(arr, dtype=torch.float32)
    target_values = target_array.view(-1)
    # compute bin edges
    edges = _compute_tensor_bin_edges(target_values, n_bins, non_parametric)
    return torch.bucketize(target_values, edges), edges


def get_stratified_train_test_split(  # noqa: WPS210
    dataset: TensorDataset,
    n_folds: int,
    non_parametric: bool = False,
    train_val: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits a TensorDataset into stratified train/test or train/val indices."""
    n_test, n_trainval, n_bins = _get_fold_params(n_folds)
    total = len(dataset)

    # validate total samples
    expected = n_trainval if train_val else n_trainval + n_test
    _validate_total_samples(total, expected, train_val)

    # stratify labels and get edges
    target_tensor = dataset.tensors[1]
    labels, edges = _stratify_tensor_labels(target_tensor, n_bins, non_parametric)
    indices = torch.arange(total)

    # perform split
    if train_val:
        n_val = n_trainval // n_folds
        train_size = (n_trainval - n_val) / total
    else:
        train_size = n_trainval / total

    train_idx, test_idx = _split_indices(indices, train_size, labels)
    train_val_indices = (train_idx, test_idx)
    return train_val_indices


def get_single_target_tensor(tensor: torch.Tensor, target_index: int) -> torch.Tensor:
    """Extract a single target column as a 2D tensor."""
    return tensor[:, target_index].unsqueeze(1)


def get_target_tensor(target_df, target_index: int | None = None) -> torch.Tensor:
    """Convert a DataFrame of targets to a tensor, optionally selecting one column."""
    tensor = torch.tensor(target_df.values, dtype=torch.float32)
    if target_index is not None:
        if tensor.ndim != 2 or tensor.shape[1] <= target_index:
            raise ValueError(
                "target_index can only be used when selecting a single target column"
            )
        tensor = get_single_target_tensor(tensor, target_index)
    return tensor


def _split_indices(
    indices: torch.Tensor, train_size: float, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits indices into train and test/validation sets."""
    train_idx, test_idx = train_test_split(
        indices, train_size=train_size, stratify=labels
    )
    return torch.tensor(train_idx), torch.tensor(test_idx)
