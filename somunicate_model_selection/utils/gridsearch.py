import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

_FOLD_PARAMS = (
    (3, 205, 600, 10),
    (4, 161, 644, 7),
    (5, 125, 680, 8),
    (6, 115, 690, 5),
)


def _get_fold_params(n_folds: int) -> tuple[int, int, int]:
    try:
        # only return the number of test samples, train/val samples, and bins
        return _FOLD_PARAMS[n_folds - 3][1:]
    except KeyError:
        raise ValueError(f"n_folds must be one of {tuple(_FOLD_PARAMS)}") from None


def _apply_pca(arr: np.ndarray) -> np.ndarray:
    """Applies PCA to reduce the array to one component."""
    return PCA(n_components=1).fit_transform(arr)


def _compute_bin_edges(
    target_values: np.ndarray, n_bins: int, non_parametric: bool
) -> np.ndarray:
    """Computes bin edges for stratification."""
    if non_parametric:
        edges = np.quantile(target_values, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-3
        edges[-1] += 1e-3
    else:
        bin_min = target_values.min() - 1e-3
        bin_max = target_values.max() + 1e-3
        edges = np.linspace(bin_min, bin_max, n_bins + 1)
    return edges


def _stratify_array_labels(
    target_data: np.ndarray, n_bins: int, non_parametric: bool
) -> np.ndarray:
    """Stratifies array labels for train/test split."""
    # ensure 2D array
    arr = target_data if target_data.ndim == 2 else target_data.reshape(-1, 1)
    # reduce to one dimension
    if arr.shape[1] > 1:
        arr = _apply_pca(arr)
    target_values = arr.ravel()
    # compute bin edges
    edges = _compute_bin_edges(target_values, n_bins, non_parametric)
    return np.digitize(target_values, bins=edges)


def _validate_total_samples(total: int, expected: int, train_val: bool) -> None:
    """
    Validates the total number of samples against the expected number.

    Args:
        total (int): The total number of samples.
        expected (int): The expected number of samples.
        train_val (bool): Whether the split is for training/validation or full dataset.

    Raises:
        ValueError: If the total number of samples does not match the expected number.
    """
    if total != expected:
        kind = "train/val" if train_val else "full"
        raise ValueError(f"Expected {expected} samples for {kind} split, got {total}")


def _perform_train_test_split(
    indices: np.ndarray, labels: np.ndarray, train_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a stratified train/test split.

    Args:
        indices (np.ndarray): Array of sample indices.
        labels (np.ndarray): Array of stratification labels.
        train_size (float): Proportion of the dataset to include in the training split.

    Returns:
        tuple[np.ndarray, np.ndarray]: Train and test indices.
    """
    return train_test_split(indices, train_size=train_size, stratify=labels)


def get_stratified_array_train_test_split(  # noqa: WPS210
    feature_data: np.ndarray,
    target_data: np.ndarray,
    n_folds: int,
    non_parametric: bool = False,
    train_val: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits arrays into stratified train/test or train/validation indices.

    Args:
        feature_data (np.ndarray): Feature data array.
        target_data (np.ndarray): Target data array.
        n_folds (int): Number of folds for cross-validation.
        non_parametric (bool, optional): Whether to use non-parametric binning. Defaults to False.
        train_val (bool, optional): Whether to split into train/validation. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Train and test/validation indices.
    """
    n_test, n_trainval, n_bins = _get_fold_params(n_folds)
    total = feature_data.shape[0]

    # validate total samples
    expected = n_trainval if train_val else n_trainval + n_test
    _validate_total_samples(total, expected, train_val)

    # stratify labels
    labels = _stratify_array_labels(target_data, n_bins, non_parametric)
    indices = np.arange(total)

    # perform split
    if train_val:
        n_val = n_trainval // n_folds
        train_size = (n_trainval - n_val) / total
    else:
        train_size = n_trainval / total

    train_idx, test_idx = _perform_train_test_split(indices, labels, train_size)
    return train_idx, test_idx


def get_pseudo_classes(
    target_data: np.ndarray, n_folds: int, non_parametric: bool = False
) -> np.ndarray:
    """
    Converts continuous target values into pseudo-classes for stratification.

    Args:
        target_data (np.ndarray): Target data array.
        n_folds (int): Number of folds for cross-validation.
        non_parametric (bool, optional): Whether to use non-parametric binning. Defaults to False.

    Returns:
        np.ndarray: Array of pseudo-classes.
    """
    # number of stratification bins equals per-fold setting
    _, _, n_bins = _get_fold_params(n_folds)
    labels = _stratify_array_labels(target_data, n_bins, non_parametric)
    # zero-based classes
    return labels - 1


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
