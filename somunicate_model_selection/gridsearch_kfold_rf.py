import argparse
import itertools
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from utils.gridsearch import get_pseudo_classes, get_stratified_array_train_test_split
from utils.loading import load_global_variables

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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Gridsearch for Random Forest models")
    parser.add_argument(
        "--subset",
        type=str,
        default=DIMENSIONS,
        help="Which subset of targets to train for. Options: 'dimensions', 'status', 'appeal', 'brand_identity', 'all'",
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
        default="./logs/gridsearch_rf",
        help="Log folder for results.",
    )
    parser.add_argument("--log_subfolder", type=str, help="Subfolder for results.")
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
    data_df = pd.read_json("./data/dummy_audio_dataset.json", orient="records")
    global_variables = load_global_variables()
    return data_df, global_variables


def prepare_target_data(
    data_df: pd.DataFrame,
    subset: str,
    target_index: int,
    global_variables: dict,
) -> tuple[str, NDArray[np.float32]]:
    """Prepare target data based on the subset.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        subset (str): Subset type.
        target_index (int): Index of the target column.
        global_variables (dict): Global variables.

    Returns:
        tuple[str, NDArray[np.float32]]: Target name and target data array.
    """
    subset_mapping = {
        DIMENSIONS: global_variables["target_list"],
        STATUS: global_variables["status_list"],
        APPEAL: global_variables["appeal_list"],
        BRAND_IDENTITY: global_variables["brand_identity_list"],
        ALL: global_variables["target_list"],
    }
    target_df = data_df[subset_mapping[subset]]
    target_data = target_df.values

    if subset == DIMENSIONS:
        target_name = target_df.columns[target_index]
        target_data = np.expand_dims(target_data[:, target_index], axis=1)
    else:
        target_name = subset

    return target_name, target_data


def prepare_feature_data(
    data_df: pd.DataFrame, global_variables: dict, include_industry: str
) -> NDArray[np.float32]:
    """Prepare feature data.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        global_variables (dict): Global variables.
        include_industry (str): Whether to include industry features.

    Returns:
        NDArray[np.float32]: Feature data array.
    """
    feature_df = data_df.drop(columns=global_variables["target_list"]).drop(
        columns=["sound"]
    )
    if include_industry == "False":
        industry_columns = [col for col in feature_df.columns if "topic" not in col]
        feature_df = feature_df.drop(columns=industry_columns)
    return feature_df.values


def perform_gridsearch(  # noqa: WPS210
    feature_data: NDArray[np.float32],
    target_data: NDArray[np.float32],
    target_name: str,
    hyperparam_dict: dict,
    folds: int,
    result_folder: str,
) -> None:
    """Perform gridsearch with k-fold cross-validation.

    Args:
        feature_data (NDArray[np.float32]): Feature data array.
        target_data (NDArray[np.float32]): Target data array.
        target_name (str): Name of the target.
        hyperparam_dict (dict): Hyperparameter dictionary.
        folds (int): Number of folds for k-fold cross-validation.
        result_folder (str): Folder to save results.
    """
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
        n_folds=folds,
        non_parametric=True,
    )

    hyperparam_combinations = list(
        itertools.product(
            *[hval for hval in hyperparam_dict.values() if isinstance(hval, Iterable)]
        )
    )
    train_val_features = feature_data[train_indices]
    train_val_targets = target_data[train_indices]
    pseudo_target_classes = get_pseudo_classes(
        train_val_targets, n_folds=folds, non_parametric=True
    )

    kf_split = StratifiedKFold(n_splits=folds, shuffle=False)
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    fold_index = 0
    model_version = 0

    for train_indices, valid_indices in kf_split.split(
        train_val_features, pseudo_target_classes
    ):
        fold_index += 1

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


def main() -> None:  # noqa: WPS210
    """Main function to execute the gridsearch."""
    args = parse_arguments()
    data_df, global_variables = load_data()
    target_name, target_data = prepare_target_data(
        data_df, args.subset, args.target_index, global_variables
    )
    feature_data = prepare_feature_data(
        data_df, global_variables, args.include_industry
    )

    result_folder = f"{args.log_folder}/{target_name}/{args.log_subfolder}/"
    os.makedirs(result_folder, exist_ok=True)

    hyperparameter_grid = HyperparameterGrid()
    hyperparam_dict: dict[str, list[Any]] = {
        N_ESTIMATORS: hyperparameter_grid.n_estimators,
        MAX_DEPTH: hyperparameter_grid.max_depth,
        MIN_SAMPLES_SPLIT: hyperparameter_grid.min_samples_split,
        MIN_SAMPLES_LEAF: hyperparameter_grid.min_samples_leaf,
        MAX_FEATURES: hyperparameter_grid.max_features,
    }

    perform_gridsearch(
        feature_data,
        target_data,
        target_name,
        hyperparam_dict,
        args.folds,
        result_folder,
    )


if __name__ == "__main__":
    main()
