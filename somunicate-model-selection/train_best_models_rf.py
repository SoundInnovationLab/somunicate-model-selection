import argparse
import json
import logging
import os
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from utils.gridsearch import get_stratified_array_train_test_split
from utils.loading import load_global_variables

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

ORIENT_RECORDS = "records"
TARGET = "target"
TARGET_LIST = "target_list"
MAX_DEPTH = "max_depth"
MAX_FEATURES = "max_features"


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

hparam_file = args.hparam_file
try:
    hyperparams = pd.read_json(hparam_file, orient=ORIENT_RECORDS).to_dict(
        orient=ORIENT_RECORDS
    )[0]
except (ValueError, FileNotFoundError) as exc:
    logger.error(f"Error loading hyperparameters: {exc}")


for key in (MAX_DEPTH, MAX_FEATURES):
    if key in hyperparams and pd.isna(hyperparams[key]):
        hyperparams[key] = None

if hyperparams[TARGET] in global_variables[TARGET_LIST]:
    subset = DIMENSIONS
else:
    subset = hyperparams[TARGET]
# load data depending on subset
data_df = pd.read_json("data/dummy_audio_dataset.json", orient=ORIENT_RECORDS)
SubsetType = Literal["dimensions", "status", "appeal", "brand_identity", "all"]
subset_mapping = {
    "dimensions": global_variables[TARGET_LIST],
    "status": global_variables["status_list"],
    "appeal": global_variables["appeal_list"],
    "brand_identity": global_variables["brand_identity_list"],
    "all": global_variables[TARGET_LIST],
}

target_df = data_df[subset_mapping[subset]]
target_data = target_df.values

# only for 'dimensions' a single target tensor is predicted
# target_name is used for logging so individual target names are needed
if subset == DIMENSIONS:
    target_idx = global_variables[TARGET_LIST].index(hyperparams[TARGET])
    target_name = global_variables[TARGET_LIST][target_idx]
    target_data = np.expand_dims(target_data[:, target_idx], axis=1)
else:
    target_name = subset


# get feature data
feature_df = data_df.drop(columns=global_variables[TARGET_LIST]).drop(columns=["sound"])
# argparser has troubles with handling boolean arguments
if args.include_industry == "False":
    industry_columns = [col for col in feature_df.columns if "topic" not in col]
    feature_df = feature_df.drop(columns=industry_columns)

feature_data = feature_df.values

result_folder = f"{args.log_folder}/{target_name}/"
os.makedirs(result_folder, exist_ok=True)
result_df = pd.DataFrame(
    columns=[
        TARGET,
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
with open(test_indices_file, "w") as test_indice_file:
    json.dump(test_indices.tolist(), test_indice_file)

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
    result_df.to_json(f"{result_folder}results.json", orient=ORIENT_RECORDS, indent=4)
