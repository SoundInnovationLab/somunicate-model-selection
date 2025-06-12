import argparse
import itertools
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from utils.gridsearch import get_pseudo_classes, get_stratified_array_train_test_split
from utils.loading import load_global_variables

warnings.filterwarnings("ignore")

DIMENSIONS = "dimensions"
STATUS = "status"
APPEAL = "appeal"
BRAND_IDENTITY = "brand_identity"
ALL = "all"


N_ESTIMATORS = "n_estimators"
MAX_DEPTH = "max_depth"
MIN_SAMPLES_SPLIT = "min_samples_split"
MIN_SAMPLES_LEAF = "min_samples_leaf"
MAX_FEATURES = "max_features"


@dataclass
class HyperparameterGrid:
    n_estimators: list[int] = field(
        default_factory=lambda: [
            100,
        ]
    )
    max_depth: list[int | None] = field(default_factory=lambda: [None])
    min_samples_split: list[int] = field(default_factory=lambda: [2])
    min_samples_leaf: list[int] = field(default_factory=lambda: [1, 2])
    max_features: list[str | None] = field(default_factory=lambda: [None])


parser = argparse.ArgumentParser(description="Gridsearch for DNN models")
parser.add_argument(
    "--subset",
    type=str,
    default=DIMENSIONS,
    help="Which subset of targets to train for. Options: 'dimensions', 'status', 'appeal', 'brand_identity', 'all'",
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
    default="./logs/gridsearch_rf",
    help="Log folder for results.",
)
parser.add_argument("--log_subfolder", type=str, help="Subfolder for results.")
# argparser has troubles with handling boolean arguments
parser.add_argument(
    "--include_industry", type=str, default="True", help="Include industry as feature."
)
args = parser.parse_args()

global_variables = load_global_variables()

# load data depending on subset
data_df = pd.read_json("data/dummy_audio_dataset.json", orient="records")
SubsetType = Literal["dimensions", "status", "appeal", "brand_identity", "all"]
subset_mapping = {
    "dimensions": global_variables["target_list"],
    "status": global_variables["status_list"],
    "appeal": global_variables["appeal_list"],
    "brand_identity": global_variables["brand_identity_list"],
    "all": global_variables["target_list"],
}

# Get target data based on subset
subset: SubsetType = args.subset  # type: ignore # tell mypy it is one of the literal types
target_df = data_df[subset_mapping[subset]]
target_data = target_df.values

# only for 'dimensions' a single target tensor is predicted
# target_name is used for logging so individual target names are needed
if args.subset == DIMENSIONS:
    target_idx = args.target_index
    target_name = target_df.columns[target_idx]
    target_data = np.expand_dims(target_data[:, target_idx], axis=1)
else:
    target_name = args.subset


# get feature data
feature_df = data_df.drop(columns=global_variables["target_list"]).drop(
    columns=["sound"]
)
# argparser has troubles with handling boolean arguments
if args.include_industry == "False":
    industry_columns = [col for col in feature_df.columns if "topic" not in col]
    feature_df = feature_df.drop(columns=industry_columns)

feature_data = feature_df.values

# prepare result saving
result_folder = f"{args.log_folder}/{target_name}/{args.log_subfolder}/"
os.makedirs(result_folder, exist_ok=True)  # Ensure the folder exists
result_df = pd.DataFrame(
    columns=[
        "model_version",
        "target",
        "fold",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "val_loss",
        "val_r2",
    ]
)
result_file_path = f"{result_folder}results.json"

train_indices, test_indices = get_stratified_array_train_test_split(
    feature_data=feature_data,
    target_data=target_data,
    n_folds=args.folds,
    non_parametric=True,
)

# Random Forest Hyperparameters
# hyperparam_dict = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [None, 10, 20],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": [None, "sqrt", "log2"],
# }

hyperparameter_grid = HyperparameterGrid()

hyperparam_dict = {
    N_ESTIMATORS: hyperparameter_grid.n_estimators,
    MAX_DEPTH: hyperparameter_grid.max_depth,
    MIN_SAMPLES_SPLIT: hyperparameter_grid.min_samples_split,
    MIN_SAMPLES_LEAF: hyperparameter_grid.min_samples_leaf,
    MAX_FEATURES: hyperparameter_grid.max_features,  # Added max_features for completeness
}

hyperparam_combinations = list(
    itertools.product(
        *[hval for hval in hyperparam_dict.values() if isinstance(hval, Iterable)]
    )
)
# stratified k-fold split needs the X and y for stratified splitting
train_val_features = feature_data[train_indices]
train_val_targets = target_data[train_indices]
# continuous targets need to be converted to pseudo classes
# (same number of classes as stratification bins in train-test)
pseudo_target_classes = get_pseudo_classes(
    train_val_targets, n_folds=args.folds, non_parametric=True
)

# go through all folds
kf_split = StratifiedKFold(n_splits=args.folds, shuffle=False)
# use sklearn mse scorer to create rmse
rmse_scorer = make_scorer(mean_squared_error, squared=False)
fold_index = 0
model_version = 0

for train_indices, valid_indices in kf_split.split(
    train_val_features, pseudo_target_classes
):
    fold_index += 1

    # do a grid search for hyperparameters
    for hparams in hyperparam_combinations:
        params_dict = dict(zip(hyperparam_dict.keys(), hparams, strict=True))

        train_features = train_val_features[train_indices]
        train_targets = train_val_targets[train_indices]

        valid_features = train_val_features[valid_indices]
        valid_targets = train_val_targets[valid_indices]

        model = RandomForestRegressor(**params_dict)
        model.fit(train_features, train_targets)

        valid_loss = rmse_scorer(model, valid_features, valid_targets)
        valid_r2 = model.score(valid_features, valid_targets)

        result_df.loc[model_version] = [
            model_version,
            target_name,
            fold_index,
            params_dict[N_ESTIMATORS],
            params_dict[MAX_DEPTH],
            params_dict[MIN_SAMPLES_SPLIT],
            params_dict[MIN_SAMPLES_LEAF],
            params_dict[MAX_FEATURES],
            valid_loss,
            valid_r2,
        ]
        model_version += 1
        result_df.to_json(result_file_path, orient="records", indent=4)
