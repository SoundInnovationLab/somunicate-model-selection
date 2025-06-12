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
from typing import Literal

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


parser = argparse.ArgumentParser(description="Gridsearch for DNN models")
parser.add_argument(
    "--subset",
    type=str,
    default=DIMENSIONS,
    help="Which subset of targets to train for. Options: all, status, appeal, brand_identity, dimensions",
)
parser.add_argument(
    "--target_index", type=int, default=0, help="Which of the 19 targets to train for."
)
parser.add_argument(
    "--folds", type=int, default=5, help="Number of folds for k-fold cross-validation."
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
    "--include_industry", type=str, default="True", help="Include industry as feature."
)
args = parser.parse_args()
global_variables = load_global_variables()


# get full feature and target data
logger.info("Loading data...")
data_df = pd.read_json("./data/dummy_audio_dataset.json", orient="records")

# Define subset types and mapping
SubsetType = Literal["dimensions", "status", "appeal", "brand_identity", "all"]
subset_mapping = {
    DIMENSIONS: global_variables["target_list"],
    STATUS: global_variables["status_list"],
    APPEAL: global_variables["appeal_list"],
    BRAND_IDENTITY: global_variables["brand_identity_list"],
    ALL: global_variables["target_list"],  # Assuming 'all' uses the full target list
}

# Get target data based on subset
subset: SubsetType = args.subset  # type: ignore # tell mypy it is one of the literal types
target_df = data_df[subset_mapping[subset]]

target_data = torch.tensor(target_df.values, dtype=torch.float32)

# Handle target tensor creation based on subset
if subset == DIMENSIONS:
    target_idx = args.target_index
    target_name = target_df.columns[target_idx]
    target_tensor = get_single_target_tensor(target_data, target_idx)
else:
    target_name = subset
    target_tensor = target_data

# create full dataset
feature_df = data_df.drop(columns=global_variables["target_list"]).drop(
    columns=["sound"]
)
# argparser has troubles with handling boolean arguments
if args.include_industry == "False":
    industry_columns = [col for col in feature_df.columns if "topic" not in col]
    feature_df = feature_df.drop(columns=industry_columns)
feature_tensor = torch.tensor(feature_df.values, dtype=torch.float32)
dataset = TensorDataset(feature_tensor, target_tensor)

# prepare result saving
result_folder = f"{args.log_folder}/{target_name}/{args.log_subfolder}/"
os.makedirs(result_folder, exist_ok=True)  # Ensure the folder exists
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
    dataset, n_folds=args.folds, non_parametric=True
)

# define gridsearch hyperparameters
# hyperparam_dict = {
#     "start_lr": [0.001, 0.0001],
#     "batch_size": [16, 32],
#     "batch_norm": [False],
#     "dropout": [0.0, 0.3, 0.5],
#     "layer_config": [
#         [128, 256],
#         [256, 128],
#         [32, 64, 128],
#         [64, 128, 64],
#         [128, 64, 32],
#         [32, 64, 128, 256],
#         [64, 128, 128, 64],
#         [256, 128, 64, 32],
#     ],
# }
grid_search_config = GridSearchConfig()
hyperparam_dict = {
    START_LR: grid_search_config.start_lr,
    BATCH_SIZE: grid_search_config.batch_size,
    BATCH_NORM: grid_search_config.batch_norm,
    DROPOUT: grid_search_config.dropout,
    LAYER_CONFIG: grid_search_config.layer_config,
}

hyperparam_combinations = list(
    itertools.product(
        *[hval for hval in hyperparam_dict.values() if isinstance(hval, Iterable)]
    )
)
# stratified k-fold split needs the X and y for stratified splitting
train_val_features = feature_tensor[train_indices]
train_val_targets = target_tensor[train_indices]
# continuous targets need to be converted to pseudo classes
# (same number of classes as stratification bins in train-test)
pseudo_target_classes = get_pseudo_classes(
    train_val_targets, n_folds=args.folds, non_parametric=True
)

# go through all folds
kf_split = StratifiedKFold(n_splits=args.folds, shuffle=False)
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
