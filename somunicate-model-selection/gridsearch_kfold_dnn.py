# load data
# per target do gridsearch on params and kfold cross validation
# save results to dataframe (append per train run and per target)
# save all runs per target in one lightning log


# imports
import argparse
import itertools
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from DNN_models import DNNRegressor
from utils.gridsearch import (
    get_pseudo_classes,
    get_single_target_tensor,
    get_stratified_train_test_split,
)
from utils.utils import load_global_variables

torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description="Gridsearch for DNN models")
parser.add_argument(
    "--subset",
    type=str,
    default="dimensions",
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
data_df = pd.read_json("./data/240918_median_data_3_features.json", orient="records")

# get target data depending on subset (one dimension, one level or all)
if args.subset == "dimensions" or args.subset == "all":
    target_df = data_df[global_variables["target_list"]]
elif args.subset == "status":
    target_df = data_df[global_variables["status_list"]]
elif args.subset == "appeal":
    target_df = data_df[global_variables["appeal_list"]]
elif args.subset == "brand_identity":
    target_df = data_df[global_variables["brand_identity_list"]]

print(target_df.columns)

target_data = torch.tensor(target_df.values, dtype=torch.float32)
# only for 'dimensions' a single target tensor is predicted
# for all other options multiple outputs are predicted
# (one communication level or all dimensions)
if args.subset == "dimensions":
    target_idx = args.target_index
    target_name = target_df.columns[target_idx]
    target_tensor = get_single_target_tensor(target_data, target_idx)
elif args.subset == "status":
    target_name = "status"
    target_tensor = target_data
elif args.subset == "appeal":
    target_name = "appeal"
    target_tensor = target_data
elif args.subset == "brand_identity":
    target_name = "brand_identity"
    target_tensor = target_data
elif args.subset == "all":
    target_name = "all"
    target_tensor = target_data

# create full dataset
feature_df = data_df.drop(columns=global_variables["target_list"]).drop(
    columns=["sound"]
)
# argparser has troubles with handling boolean arguments
if args.include_industry == "False":
    industry_columns = [col for col in feature_df.columns if "topic" not in col]
    feature_df = feature_df.drop(columns=industry_columns)
print(feature_df.columns)
feature_tensor = torch.tensor(feature_df.values, dtype=torch.float32)
dataset = TensorDataset(feature_tensor, target_tensor)

# prepare result saving
print(f"Training for target {target_name}")
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
print("Train+Validation", len(train_indices), "Test", len(test_indices))

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
hyperparam_dict = {
    "start_lr": [0.001, 0.0001],
    "batch_size": [32],
    "batch_norm": [False],
    "dropout": [0.5],
    "layer_config": [
        [128, 256],
    ],
}

hyperparam_combinations = list(itertools.product(*hyperparam_dict.values()))
print(len(hyperparam_combinations), "combinations")

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
    print(
        "Fold",
        fold_index,
        "Train",
        len(train_indices),
        "Validation",
        len(valid_indices),
    )

    # make sure that batch size modulo 16 or 32 is never 1
    # r^2 metric would return null in this case
    if len(train_indices) % 16 == 1 or len(train_indices) % 32 == 1:
        train_indices = train_indices[:-1]
    if len(valid_indices) % 16 == 1 or len(valid_indices) % 32 == 1:
        valid_indices = valid_indices[:-1]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    valid_set = torch.utils.data.Subset(dataset, valid_indices)

    # do a grid search for hyperparameters
    for hparams in hyperparam_combinations:
        params = dict(zip(hyperparam_dict.keys(), hparams, strict=True))

        train_loader = DataLoader(
            train_set, batch_size=params["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=params["batch_size"], shuffle=False
        )

        input_dim = train_loader.dataset[0][0].shape[0]
        output_dim = train_loader.dataset[0][1].shape[0]

        print(input_dim, output_dim)

        model = DNNRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=params["layer_config"],
            use_batch_norm=params["batch_norm"],
            dropout_prob=params["dropout"],
            learning_rate=params["start_lr"],
            lr_scheduler=True,
            weight_decay=True,
            target_name=target_name,
            loss_type="rmse",
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
            params["start_lr"],
            params["batch_size"],
            params["batch_norm"],
            params["dropout"],
            params["layer_config"],
            valid_loss,
            valid_r2,
        ]
        model_version += 1
        result_df.to_json(result_file_path, orient="records", indent=4)
