import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def get_stratified_array_train_test_split(
    feature_data: np.ndarray,
    target_data: np.ndarray,
    n_folds: int,
    n_stratification_bins: int = 10,
    non_parametric: bool = False,
    train_val=False,
):
    """
    Splits the dataset into train, and test sets using numpy arrays.
    Args:
        feature_data (np.ndarray): The feature data.
        target_data (np.ndarray): The target data.
        n_folds (int): Number of k-folds to use for stratified sampling. Train size and number of test and train samples depend on the folds.
        n_stratification_bins (int): Number of bins to use for stratified sampling.
            If None, stratified sampling is not used.
        non_parametric (bool): If True, the target space is divided into bins according to quartiles.
        train_val (bool): If True, the dataset is split into train and validation sets with according numbers.
    Returns:
        tuple: A tuple containing the train, validation, and test sets.
    """

    assert n_folds in [3, 4, 5, 6]

    if n_folds == 3:
        n_test_samples = 205
        n_trainval_samples = 600
        n_stratification_bins = 10
    elif n_folds == 4:
        n_test_samples = 161
        n_trainval_samples = 644
        n_stratification_bins = 7
    elif n_folds == 5:
        n_test_samples = 125
        n_trainval_samples = 680
        n_stratification_bins = 8
    elif n_folds == 6:
        n_test_samples = 115
        n_trainval_samples = 690
        n_stratification_bins = 5

    total_samples = feature_data.shape[0]

    if not train_val:
        assert total_samples == 805
    else:
        assert total_samples == n_trainval_samples
        n_val_samples = n_trainval_samples // n_folds

    # Perform PCA if target has more than one dimension
    if target_data.shape[1] > 1:
        pca = PCA(n_components=1)
        target_pca = pca.fit_transform(target_data)
    else:
        target_pca = target_data

    # Define bin edges based on the stratification strategy
    if non_parametric:
        # Divide target space into bins per according to quartiles for skewed distributions
        bin_edges = np.quantile(
            target_pca, np.linspace(0, 1, n_stratification_bins + 1)
        )
        # Add a small value to the first and last bin edge to include the minimum/maximum value
        bin_edges[0] -= 1e-3
        bin_edges[-1] += 1e-3
    else:
        # Divide target space into equidistant bins
        min_val = np.min(target_pca) - 1e-3
        max_val = np.max(target_pca) + 1e-3
        bin_edges = np.linspace(min_val, max_val, n_stratification_bins + 1)

    # Assign each target value to a bin
    target_binned = np.digitize(target_pca, bins=bin_edges)

    # Composite bins for multidimensional targets (if needed)
    composite_bins = target_binned[:, 0]
    if target_binned.shape[1] > 1:
        for i in range(1, target_binned.shape[1]):
            composite_bins *= n_stratification_bins + 1
            composite_bins += target_binned[:, i]

    # Train-Test split based on stratification
    if not train_val:
        train_indices, test_indices = train_test_split(
            np.arange(total_samples),
            train_size=n_trainval_samples / total_samples,
            stratify=composite_bins,
        )
        assert len(train_indices) == n_trainval_samples
        assert len(test_indices) == n_test_samples
        return train_indices, test_indices
    else:
        train_indices, val_indices = train_test_split(
            np.arange(total_samples),
            train_size=(n_trainval_samples - n_val_samples) / total_samples,
            stratify=composite_bins,
        )
        assert len(train_indices) == n_trainval_samples - n_val_samples
        assert len(val_indices) == n_val_samples
        return train_indices, val_indices


def get_pseudo_classes(
    target_data: np.ndarray, n_folds: int, non_parametric: bool = False
):
    """
    Converts continuous target data into pseudo-classes for StratifiedKFold sampling.
    The target data is divided into bins according to the number of folds.

    Args:
        target_data (np.ndarray): The target data to convert.
        n_folds (int): Number of folds to use for stratified sampling.
        non_parametric (bool): If True, the target space is divided into bins according to quartiles.

    Returns:
        np.ndarray: The pseudo-classes.
    """

    assert n_folds in [3, 4, 5, 6]

    # Determine the number of classes based on the number of folds
    if n_folds == 3:
        n_classes = 10
    elif n_folds == 4:
        n_classes = 7
    elif n_folds == 5:
        n_classes = 8
    elif n_folds == 6:
        n_classes = 5

    # Perform PCA if the target data has more than one dimension
    if target_data.shape[1] > 1:
        pca = PCA(n_components=1)
        target_pca = pca.fit_transform(target_data)
    else:
        target_pca = target_data

    target_binned = np.zeros_like(target_pca, dtype=np.int64)

    # Divide the target space into bins based on the strategy
    if non_parametric:
        # Divide the target space into bins according to quartiles (for skewed distributions)
        bin_edges = np.quantile(target_pca, np.linspace(0, 1, n_classes + 1))
        # Add a small value to the first and last bin edge to include the minimum/maximum value
        bin_edges[0] -= 1e-3
        bin_edges[-1] += 1e-3
    else:
        # Equidistant binning using min and max
        min_val = np.min(target_pca) - 1e-3
        max_val = np.max(target_pca) + 1e-3
        bin_edges = np.linspace(min_val, max_val, n_classes + 1)

    # Assign each target value to a bin
    for i in range(target_pca.shape[1]):
        target_binned[:, i] = np.digitize(target_pca[:, i], bins=bin_edges) - 1

    return target_binned


def get_stratified_train_test_split(
    dataset: TensorDataset,
    n_folds: int,
    n_stratification_bins: int = 10,
    non_parametric: bool = False,
    train_val=False,
    return_edges=False,
):
    """
    Splits the dataset into train, and test sets.
    Args:
        dataset (TensorDataset): The dataset to split.
        n_folds (int): Number of k-folds to use for stratified sampling. Train size and number of test and train samples depends on the folds.
        n_stratification_bins (int): Number of bins to use for stratified sampling.
            If None, stratified sampling is not used.
        non_parametric (bool): If True, the target space is divided into bins according to quartiles.
        train_val (bool): If True, the dataset is split into train and validation sets with according numbers.
    Returns:
        tuple: A tuple containing the train, validation, and test sets.
    """

    assert n_folds in [3, 4, 5, 6]
    if n_folds == 3:
        n_test_samples = 205
        n_trainval_samples = 600
        n_stratification_bins = 10
    elif n_folds == 4:
        n_test_samples = 161
        n_trainval_samples = 644
        n_stratification_bins = 7
    elif n_folds == 5:
        n_test_samples = 125
        n_trainval_samples = 680
        n_stratification_bins = 8
    elif n_folds == 6:
        n_test_samples = 115
        n_trainval_samples = 690
        n_stratification_bins = 5

    if not train_val:
        assert len(dataset) == 805
    else:
        assert len(dataset) == n_trainval_samples
        n_val_samples = n_trainval_samples / n_folds

    target_tensors = dataset.tensors[1]
    if target_tensors.shape[1] > 1:
        pca = PCA(n_components=1)
        target_pca_tensor = torch.tensor(pca.fit_transform(target_tensors))
    else:
        target_pca_tensor = target_tensors

    if non_parametric:
        # divide target space into bins per according to quartiles -> for skewed distributions
        bin_edges = torch.tensor(
            np.quantile(
                target_pca_tensor.numpy(), np.linspace(0, 1, n_stratification_bins + 1)
            )
        )
        # add a small value to the first and last bin edge to include the minimum/maximum value
        bin_edges[0] -= 1e-3
        bin_edges[-1] += 1e-3
    else:
        # divide target space into bins per according to min and max with equidistant binning
        min_val = target_pca_tensor.min().item() - 1e-3
        max_val = target_pca_tensor.max().item() + 1e-3
        bin_edges = torch.linspace(
            min_val, max_val, n_stratification_bins + 1, dtype=torch.float32
        )

    target_pca_binned = torch.zeros_like(target_pca_tensor, dtype=torch.int64)
    for i in range(target_pca_tensor.shape[1]):
        target_pca_binned[:, i] = torch.bucketize(
            target_pca_tensor[:, i].contiguous(), bin_edges
        )
    # print count of each bin
    # combine bins into a single number
    composite_bins = target_pca_binned[:, 0].clone()
    for i in range(1, target_pca_binned.shape[1]):
        composite_bins *= n_stratification_bins + 1
        composite_bins += target_pca_binned[:, i]

    if not train_val:
        train_indices, test_indices = train_test_split(
            torch.arange(len(target_tensors)),
            train_size=n_trainval_samples / len(dataset),
            stratify=composite_bins,
        )
        assert len(train_indices) == n_trainval_samples
        assert len(test_indices) == n_test_samples
        if return_edges:
            return train_indices, test_indices, bin_edges
        else:
            return train_indices, test_indices
    else:
        train_indices, val_indices = train_test_split(
            torch.arange(len(target_tensors)),
            train_size=(n_trainval_samples - n_val_samples) / len(dataset),
            stratify=composite_bins,
        )
        assert len(train_indices) == n_trainval_samples - n_val_samples
        assert len(val_indices) == n_val_samples
        if return_edges:
            return train_indices, val_indices, bin_edges
        else:
            return train_indices, val_indices


def get_single_target_tensor(target_tensor, target_index):
    target_tensor = target_tensor[:, target_index]
    target_tensor = target_tensor.unsqueeze(1)
    return target_tensor


def get_target_tensor(target_df, target_index: int = None):
    target_data = torch.tensor(target_df.values, dtype=torch.float32)
    # only applies to "dimension" subset where only one target is used
    if target_index is not None:
        assert target_data.shape[1] == 19, (
            'target_index can only be used for "dimensions" subset meaning single target data'
        )
        target_data = get_single_target_tensor(target_data, target_index)
    return target_data
