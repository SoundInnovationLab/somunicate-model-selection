import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

from utils.gridsearch import get_stratified_array_train_test_split
from utils.utils import load_global_variables

global_variables = load_global_variables()

parser = argparse.ArgumentParser(description="Gridsearch for Random Forest Models")
parser.add_argument("--hparam_file", type=str, help="File with best hyperparameters")
parser.add_argument(
    "--log_folder",
    type=str,
    default="logs/best_models_rf",
    help="Log folder for results.",
)
parser.add_argument(
    "--n_folds",
    type=int,
    default=5,
    help="Number of folds for k-fold cross-validation.",
)
parser.add_argument(
    "--n_experiments", type=int, default=1, help="Number of repeated experiments."
)
parser.add_argument(
    "--include_industry", type=str, default="True", help="Include industry as feature."
)

args = parser.parse_args()

try:
    hyperparams = pd.read_json(args.hparam_file, orient="records").to_dict(
        orient="records"
    )[0]
    # make sure 'max_depth' and 'n_estimators' have None if they are null or N an
    for key, value in hyperparams.items():
        if key == "max_depth" or key == "max_features":
            # check for nan
            if value != value:
                print(key, value)
                # set to None
                hyperparams[key] = None
                print(key, hyperparams[key])
except (ValueError, FileNotFoundError) as e:
    print(f"Could not read hyperparameter file {args.hparam_file}")
    exit()


if hyperparams["target"] in global_variables["target_list"]:
    subset = "dimensions"
else:
    subset = hyperparams["target"]
# load data depending on subset
data_df = pd.read_json("data/240918_median_data_3_features.json", orient="records")
if subset == "dimensions" or subset == "all":
    target_df = data_df[global_variables["target_list"]]
elif subset == "status":
    target_df = data_df[global_variables["status_list"]]
elif subset == "appeal":
    target_df = data_df[global_variables["appeal_list"]]
elif subset == "brand_identity":
    target_df = data_df[global_variables["brand_identity_list"]]

target_data = target_df.values

# only for 'dimensions' a single target tensor is predicted
# target_name is used for logging so individual target names are needed
if subset == "dimensions":
    target_idx = global_variables["target_list"].index(hyperparams["target"])
    target_name = global_variables["target_list"][target_idx]
    target_data = np.expand_dims(target_data[:, target_idx], axis=1)
else:
    target_name = subset


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

result_folder = f"{args.log_folder}/{target_name}/"
os.makedirs(result_folder, exist_ok=True)
result_df = pd.DataFrame(
    columns=[
        "target",
        "folds",
        "experiment_id",
        "test_loss",
        "test_r2",
    ]
)

train_indices, test_indices = get_stratified_array_train_test_split(
    feature_data=feature_data,
    target_data=target_data,
    n_folds=args.n_folds,
    non_parametric=True,
)
# save test indices to ensure all experiments are comparable
test_indices_file = f"{result_folder}test_indices.json"
with open(test_indices_file, "w") as f:
    json.dump(test_indices.tolist(), f)

train_features = feature_data[train_indices]
train_targets = target_data[train_indices]
test_features = feature_data[test_indices]
test_targets = target_data[test_indices]

rmse_scorer = make_scorer(mean_squared_error, squared=False)

for e_idx in range(args.n_experiments):
    model = RandomForestRegressor(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        min_samples_split=hyperparams["min_samples_split"],
        min_samples_leaf=hyperparams["min_samples_leaf"],
        max_features=hyperparams["max_features"],
    )
    model.fit(train_features, train_targets)
    # save model
    joblib.dump(model, f"{result_folder}model_{e_idx}.pkl")

    test_loss = rmse_scorer(model, test_features, test_targets)
    test_r2 = model.score(test_features, test_targets)

    result_df.loc[e_idx] = [target_name, args.n_folds, e_idx, test_loss, test_r2]
    result_df.to_json(f"{result_folder}results.json", orient="records", indent=4)
