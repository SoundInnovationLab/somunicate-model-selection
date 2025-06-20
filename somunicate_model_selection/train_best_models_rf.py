import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
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


def process_hyperparameters(hyperparams: dict) -> dict:
    """
    Process hyperparameters to replace null values.
    Args:
        hyperparams (dict): Dictionary of hyperparameters.
    Returns:
        dict: Processed hyperparameters with None for null values.
    """
    return {
        key: (None if pd.isnull(hparam_value) else hparam_value)
        for key, hparam_value in hyperparams.items()
    }


def load_hyperparameters(hparam_file: str) -> dict:
    """Load hyperparameters from a JSON file and handle null values.

    Args:
        hparam_file (str): Path to the hyperparameter file.

    Returns:
        dict: Dictionary containing hyperparameters with nulls converted to None.

    Raises:
        ValueError: If the file cannot be loaded.
    """

    try:
        raw_hyperparams = pd.read_json(hparam_file, orient=ORIENT_RECORDS).to_dict(
            orient=ORIENT_RECORDS
        )[0]
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Error loading hyperparameters: {exc}")
        raise

    return process_hyperparameters(raw_hyperparams)


def prepare_target_data(
    data_df: pd.DataFrame, hyperparams: dict, global_variables: dict
) -> tuple[str, NDArray[np.float32]]:
    """Prepare target data based on hyperparameters.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        hyperparams (dict): Dictionary of hyperparameters.
        global_variables (dict): Global variables.

    Returns:
        tuple[str, NDArray[np.float32]]: Target name and target data array.
    """
    subset = (
        DIMENSIONS
        if hyperparams[TARGET] in global_variables[TARGET_LIST]
        else hyperparams[TARGET]
    )

    subset_mapping = {
        DIMENSIONS: global_variables["target_list"],
        STATUS: global_variables["status_list"],
        APPEAL: global_variables["appeal_list"],
        BRAND_IDENTITY: global_variables["brand_identity_list"],
        ALL: global_variables["target_list"],
    }
    target_data = data_df[subset_mapping[subset]].values

    if subset == DIMENSIONS:
        target_idx = global_variables[TARGET_LIST].index(hyperparams[TARGET])
        target_name = global_variables[TARGET_LIST][target_idx]
        target_data = np.expand_dims(target_data[:, target_idx], axis=1)
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
    feature_df = data_df.drop(columns=global_variables[TARGET_LIST]).drop(
        columns=["sound"]
    )
    if include_industry == "False":
        industry_columns = [col for col in feature_df.columns if "topic" not in col]
        feature_df = feature_df.drop(columns=industry_columns)
    return feature_df.values


def train_and_evaluate_model(
    train_features: NDArray[np.float32],
    train_targets: NDArray[np.float32],
    test_features: NDArray[np.float32],
    test_targets: NDArray[np.float32],
    hyperparams: dict,
    result_folder: str,
    e_idx: int,
    n_folds: int,
) -> None:
    """Train and evaluate the Random Forest model.

    Args:
        train_features (NDArray[np.float32]): Training feature data.
        train_targets (NDArray[np.float32]): Training target data.
        test_features (NDArray[np.float32]): Test feature data.
        test_targets (NDArray[np.float32]): Test target data.
        hyperparams (dict): Dictionary of hyperparameters.
        target_name (str): Name of the target.
        result_folder (str): Folder to save results.
        e_idx (int): Experiment index.
        n_folds (int): Number of folds for cross-validation.
    """
    model = RandomForestRegressor(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        min_samples_split=hyperparams["min_samples_split"],
        min_samples_leaf=hyperparams["min_samples_leaf"],
        max_features=hyperparams["max_features"],
    )
    model.fit(train_features, train_targets)
    joblib.dump(model, f"{result_folder}model_{e_idx}.pkl")

    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    test_loss = rmse_scorer(model, test_features, test_targets)
    test_r2 = model.score(test_features, test_targets)

    result_df = pd.DataFrame(
        [[hyperparams["target"], n_folds, e_idx, test_loss, test_r2]],
        columns=[TARGET, "folds", "experiment_id", "test_loss", "test_r2"],
    )
    result_df.to_json(f"{result_folder}results.json", orient=ORIENT_RECORDS, indent=4)


def main() -> None:  # noqa: WPS210
    """Main function to train the best Random Forest model."""
    args = parser.parse_args()
    hyperparams = load_hyperparameters(args.hparam_file)
    data_df = pd.read_json("data/dummy_audio_dataset.json", orient=ORIENT_RECORDS)
    global_variables = load_global_variables()

    target_name, target_data = prepare_target_data(
        data_df, hyperparams, global_variables
    )
    feature_data = prepare_feature_data(
        data_df, global_variables, args.include_industry
    )

    result_folder = f"{args.log_folder}/{target_name}/"
    os.makedirs(result_folder, exist_ok=True)

    train_indices, test_indices = get_stratified_array_train_test_split(
        feature_data=feature_data,
        target_data=target_data,
        n_folds=args.n_folds,
        non_parametric=True,
    )
    train_features, train_targets = (
        feature_data[train_indices],
        target_data[train_indices],
    )
    test_features, test_targets = feature_data[test_indices], target_data[test_indices]

    for e_idx in range(args.n_experiments):
        train_and_evaluate_model(
            train_features,
            train_targets,
            test_features,
            test_targets,
            hyperparams,
            result_folder,
            e_idx,
            args.n_folds,
        )


if __name__ == "__main__":
    main()
