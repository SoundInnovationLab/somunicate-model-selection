# load data
# per target do gridsearch on params and kfold cross validation
# save results to dataframe (append per train run and per target)
# save all runs per target in one lightning log

# imports
import argparse
import itertools
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from dnn_models import DNNRegressor, LossType, ModelConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from utils.gridsearch import (
    get_pseudo_classes,
    get_single_target_tensor,
)
from utils.gridsearch_tensor import get_stratified_train_test_split
from utils.loading import load_global_variables

# for gridsearch and model selection medium precision is sufficient
# this is a trade-off between performance and training time
torch.set_float32_matmul_precision("medium")

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DIMENSIONS = "dimensions"
STATUS = "status"
APPEAL = "appeal"
BRAND_IDENTITY = "brand_identity"
ALL = "all"

START_LR = "start_lr"
BATCH_SIZE = "batch_size"
BATCH_NORM = "batch_norm"
DROPOUT = "dropout"
LAYER_CONFIG = "layer_config"


@dataclass
class GridSearchConfig:
    start_lr: list[float] = field(default_factory=lambda: [0.001, 0.0001])
    batch_size: list[int] = field(default_factory=lambda: [32])
    batch_norm: list[bool] = field(default_factory=lambda: [False])
    dropout: list[float] = field(default_factory=lambda: [0.5])
    layer_config: list[list[int]] = field(default_factory=lambda: [[128, 256]])


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Gridsearch for DNN models")
    parser.add_argument(
        "--subset",
        type=str,
        default=DIMENSIONS,
        help="Which subset of targets to train for. Options: all, status, appeal, brand_identity, dimensions",
    )
    parser.add_argument(
        "--target_index",
        type=int,
        default=0,
        help="Which of the 19 targets to train for.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross-validation.",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="./logs/gridsearch_dnn",
        help="Log folder for results.",
    )
    parser.add_argument(
        "--log_subfolder", type=str, default="test", help="Subfolder for results."
    )
    # argparser has troubles with handling boolean arguments
    parser.add_argument(
        "--include_industry",
        type=str,
        default="True",
        help="Include industry as feature.",
    )
    return parser.parse_args()


def load_data() -> tuple[Any, Any]:
    """Load the dataset and global variables.

    Returns:
        tuple[Any, Any]: Dataframe containing the dataset and global variables.
    """
    logger.info("Loading data...")
    data_df = pd.read_json("./data/dummy_audio_dataset.json", orient="records")
    global_variables = load_global_variables()
    return data_df, global_variables


def prepare_target_data(
    data_df: pd.DataFrame,
    subset: Literal["dimensions", "status", "appeal", "brand_identity", "all"],
    target_index: int,
    global_variables: dict[str, list[str]],
) -> tuple[str, torch.Tensor]:
    """Prepare target data based on the subset.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        subset (Literal): Subset type.
        target_index (int): Index of the target column.
        global_variables (dict[str, list[str]]): Global variables.

    Returns:
        tuple[str, torch.Tensor]: Target name and target tensor.
    """
    subset_mapping = {
        DIMENSIONS: global_variables["target_list"],
        STATUS: global_variables["status_list"],
        APPEAL: global_variables["appeal_list"],
        BRAND_IDENTITY: global_variables["brand_identity_list"],
        ALL: global_variables["target_list"],
    }
    target_df = data_df[subset_mapping[subset]]
    target_data = torch.tensor(target_df.values, dtype=torch.float32)

    if subset == DIMENSIONS:
        target_name = target_df.columns[target_index]
        target_tensor = get_single_target_tensor(target_data, target_index)
    else:
        target_name = subset
        target_tensor = target_data

    return target_name, target_tensor


def prepare_feature_data(
    data_df: pd.DataFrame, global_variables: dict[str, list[str]], include_industry: str
) -> torch.Tensor:
    """Prepare feature data.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        global_variables (dict[str, list[str]]): Global variables.
        include_industry (str): Whether to include industry features.

    Returns:
        torch.Tensor: Feature tensor.
    """
    feature_df = data_df.drop(columns=global_variables["target_list"]).drop(
        columns=["sound"]
    )
    if include_industry == "False":
        industry_columns = [col for col in feature_df.columns if "topic" not in col]
        feature_df = feature_df.drop(columns=industry_columns)
    return torch.tensor(feature_df.values, dtype=torch.float32)


def perform_gridsearch(  # noqa: WPS210
    dataset: TensorDataset,
    target_name: str,
    hyperparam_dict: dict[str, list[Any]],
    folds: int,
    result_folder: str,
) -> None:
    """Perform gridsearch with k-fold cross-validation.

    Args:
        dataset (TensorDataset): Dataset containing features and targets.
        target_name (str): Name of the target.
        hyperparam_dict (dict[str, list]): Hyperparameter dictionary.
        folds (int): Number of folds for k-fold cross-validation.
        result_folder (str): Folder to save results.
    """
    # prepare result saving
    result_df = pd.DataFrame(
        columns=[
            "model_version",
            "target",
            "fold",
            "start_lr",
            "batch_size",
            "batch_norm",
            "dropout",
            "layer_config",
            "val_loss",
            "val_r2",
        ]
    )
    result_file_path = f"{result_folder}results.json"

    # get test set
    train_indices, test_indices = get_stratified_train_test_split(
        dataset, n_folds=folds, non_parametric=True
    )

    hyperparam_combinations = list(
        itertools.product(
            *[hval for hval in hyperparam_dict.values() if isinstance(hval, Iterable)]
        )
    )
    # stratified k-fold split needs the X and y for stratified splitting
    train_val_features = dataset.tensors[0][train_indices]
    train_val_targets = dataset.tensors[1][train_indices]
    # continuous targets need to be converted to pseudo classes
    # (same number of classes as stratification bins in train-test)
    pseudo_target_classes = get_pseudo_classes(
        train_val_targets, n_folds=folds, non_parametric=True
    )

    # go through all folds
    kf_split = StratifiedKFold(n_splits=folds, shuffle=False)
    fold_index = 0
    model_version = 0

    for train_indices, valid_indices in kf_split.split(
        train_val_features, pseudo_target_classes
    ):
        fold_index += 1
        logger.info(f"Starting fold {fold_index}")

        # make sure that batch size modulo 16 or 32 is never 1
        # r^2 metric would return null in this case
        batch_sizes = [16, 32]
        needs_trimming = any(len(train_indices) % bs == 1 for bs in batch_sizes)
        if needs_trimming:
            train_indices = train_indices[:-1]

        needs_trimming = any(len(valid_indices) % bs == 1 for bs in batch_sizes)
        if needs_trimming:
            valid_indices = valid_indices[:-1]

        train_set = torch.utils.data.Subset(dataset, train_indices)
        valid_set = torch.utils.data.Subset(dataset, valid_indices)

        # do a grid search for hyperparameters
        for hparams in hyperparam_combinations:
            param_dict = dict(zip(hyperparam_dict.keys(), hparams, strict=True))

            train_loader = DataLoader(
                train_set, batch_size=param_dict[BATCH_SIZE], shuffle=True
            )
            valid_loader = DataLoader(
                valid_set, batch_size=param_dict[BATCH_SIZE], shuffle=False
            )

            sample_batch = next(iter(train_loader))
            input_tensor, target_tensor = sample_batch[0], sample_batch[1]

            input_dim = input_tensor.shape[1]
            output_dim = target_tensor.shape[1]

            model_config = ModelConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                use_batch_norm=param_dict[BATCH_NORM],
                dropout_prob=param_dict[DROPOUT],
                learning_rate=param_dict[START_LR],
                lr_scheduler=True,
                weight_decay=True,
                target_name=target_name,
                loss_type=LossType.RMSE,  # Use the Enum
            )

            model = DNNRegressor(
                args=model_config,
                hidden_dims=param_dict[LAYER_CONFIG],
            )

            trainer = pl.Trainer(
                max_epochs=2,
                enable_checkpointing=False,
                accelerator="gpu",
                devices=1,
                logger=False,
            )
            trainer.fit(model, train_loader, valid_loader)

            valid_loss = trainer.callback_metrics.get("valid_loss_epoch", None).item()
            valid_r2 = trainer.callback_metrics.get("valid_r2_epoch", None).item()

            result_df.loc[model_version] = [
                model_version,
                target_name,
                fold_index,
                param_dict[START_LR],
                param_dict[BATCH_SIZE],
                param_dict[BATCH_NORM],
                param_dict[DROPOUT],
                param_dict[LAYER_CONFIG],
                valid_loss,
                valid_r2,
            ]
            model_version += 1
            result_df.to_json(result_file_path, orient="records", indent=4)


def main() -> None:  # noqa: WPS210
    """Main function to execute the gridsearch."""
    args = parse_arguments()
    data_df, global_variables = load_data()
    target_name, target_tensor = prepare_target_data(
        data_df, args.subset, args.target_index, global_variables
    )
    feature_tensor = prepare_feature_data(
        data_df, global_variables, args.include_industry
    )

    dataset = TensorDataset(feature_tensor, target_tensor)
    result_folder = f"{args.log_folder}/{target_name}/{args.log_subfolder}/"
    os.makedirs(result_folder, exist_ok=True)

    grid_search_config = GridSearchConfig()
    hyperparam_dict: dict[str, list[Any]] = {
        START_LR: grid_search_config.start_lr,
        BATCH_SIZE: grid_search_config.batch_size,
        BATCH_NORM: grid_search_config.batch_norm,
        DROPOUT: grid_search_config.dropout,
        LAYER_CONFIG: grid_search_config.layer_config,
    }

    perform_gridsearch(dataset, target_name, hyperparam_dict, args.folds, result_folder)


if __name__ == "__main__":
    main()
