import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import StratifiedKFold

from utils.gridsearch import get_pseudo_classes, get_stratified_array_train_test_split
from utils.utils import load_global_variables

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Gridsearch for DNN models")
parser.add_argument(
    "--subset",
    type=str,
    default="all",
    help="Which subset of targets to train for. Options: dimensions, status, appeal, brand_identity, all",
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
data_df = pd.read_json("data/240918_median_data_3_features.json", orient="records")
if args.subset == "dimensions" or args.subset == "all":
    target_df = data_df[global_variables["target_list"]]
elif args.subset == "status":
    target_df = data_df[global_variables["status_list"]]
elif args.subset == "appeal":
    target_df = data_df[global_variables["appeal_list"]]
elif args.subset == "brand_identity":
    target_df = data_df[global_variables["brand_identity_list"]]

target_data = target_df.values

# only for 'dimensions' a single target tensor is predicted
# target_name is used for logging so individual target names are needed
if args.subset == "dimensions":
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
print("Feature Shape: ", feature_data.shape, "\nTarget Shape: ", target_data.shape)

# prepare result saving
print(f"Training for target {target_name}")
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

hyperparam_dict = {
    "n_estimators": [100],
    "max_depth": [None, 10],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": [None, "log2"],
}

hyperparam_combinations = list(itertools.product(*hyperparam_dict.values()))
print(len(hyperparam_combinations), "combinations")

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
    print(
        "Fold",
        fold_index,
        "Train",
        len(train_indices),
        "K-fold Validation",
        len(valid_indices),
    )

    # do a grid search for hyperparameters
    for hparams in hyperparam_combinations:
        params = dict(zip(hyperparam_dict.keys(), hparams, strict=True))

        train_features = train_val_features[train_indices]
        train_targets = train_val_targets[train_indices]

        valid_features = train_val_features[valid_indices]
        valid_targets = train_val_targets[valid_indices]

        model = RandomForestRegressor(**params)
        model.fit(train_features, train_targets)

        valid_loss = rmse_scorer(model, valid_features, valid_targets)
        valid_r2 = model.score(valid_features, valid_targets)

        result_df.loc[model_version] = [
            model_version,
            target_name,
            fold_index,
            params["n_estimators"],
            params["max_depth"],
            params["min_samples_split"],
            params["min_samples_leaf"],
            params["max_features"],
            valid_loss,
            valid_r2,
        ]
        model_version += 1
        result_df.to_json(result_file_path, orient="records", indent=4)
