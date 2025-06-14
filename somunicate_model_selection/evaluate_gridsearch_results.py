import argparse
import os
from dataclasses import dataclass, field

import pandas as pd
from utils.evaluation import (
    average_over_hparam_combinations,
    find_best_hparams,
    save_best_hparams_df,
)
from utils.loading import load_df


@dataclass
class RFHyperparameterGrid:
    n_estimators: list[int] = field(
        default_factory=lambda: [
            100,
        ]
    )
    max_depth: list[int | None] = field(default_factory=lambda: [None])
    min_samples_split: list[int] = field(default_factory=lambda: [2])
    min_samples_leaf: list[int] = field(default_factory=lambda: [1, 2])
    max_features: list[str | None] = field(default_factory=lambda: [None])


@dataclass
class DNNHyperparameterGrid:
    start_lr: list[float] = field(default_factory=lambda: [0.001, 0.0001])
    batch_size: list[int] = field(default_factory=lambda: [32])
    batch_norm: list[bool] = field(default_factory=lambda: [False])
    dropout: list[float] = field(default_factory=lambda: [0.5])
    layer_config: list[list[int]] = field(default_factory=lambda: [[128, 256]])


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing log_dir and learner.
    """
    parser = argparse.ArgumentParser(description="Evaluate gridsearch results")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/gridsearch_rf",
        help="Path to the log directory",
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="dnn",
        help="Deep Neural Network (dnn), Random Forest (rf)",
    )
    return parser.parse_args()


def get_hyperparam_dict(args: argparse.Namespace) -> dict[str, list]:
    """Retrieve the hyperparameter dictionary based on the learner.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, List]: Hyperparameter dictionary for the specified learner.

    Raises:
        ValueError: If the learner type is unknown.
    """
    if args.learner == "dnn":
        hyperparam_grid_rf = DNNHyperparameterGrid()
        return {
            "start_lr": hyperparam_grid_rf.start_lr,
            "batch_size": hyperparam_grid_rf.batch_size,
            "batch_norm": hyperparam_grid_rf.batch_norm,
            "dropout": hyperparam_grid_rf.dropout,
            "layer_config": hyperparam_grid_rf.layer_config,
        }
    elif args.learner == "rf":
        hyperparam_grid_dnn = RFHyperparameterGrid()
        return {
            "n_estimators": hyperparam_grid_dnn.n_estimators,
            "max_depth": hyperparam_grid_dnn.max_depth,
            "min_samples_split": hyperparam_grid_dnn.min_samples_split,
            "min_samples_leaf": hyperparam_grid_dnn.min_samples_leaf,
            "max_features": hyperparam_grid_dnn.max_features,
        }
    else:
        raise ValueError(f"Unknown learner type: {args.learner}")


def process_results(
    log_dir: str, hyperparam_dict: dict[str, list], learner: str
) -> None:
    """Process results in the log directory.

    Args:
        log_dir (str): Path to the log directory.
        hyperparam_dict (Dict[str, List]): Hyperparameter dictionary.
        learner (str): Type of learner (e.g., 'dnn', 'rf').
    """
    for subdir, _, files in os.walk(log_dir):
        for result_file in files:
            if result_file == "results.json":
                df = load_df(subdir, result_file)
                average_df = average_over_hparam_combinations(
                    df, hyperparam_dict, learner
                )
                save_average_results(subdir, average_df)
                save_best_model_hparams(subdir, average_df)


def save_average_results(log_folder: str, average_df: pd.DataFrame) -> None:
    """Save averaged results to a JSON file.

    Args:
        log_folder (str): Path to the log folder.
        average_df: DataFrame containing averaged results.
    """
    avg_results_path = f"{log_folder}/fold_average_results.json"
    if not os.path.exists(avg_results_path):
        average_df.to_json(avg_results_path, orient="records", indent=4)


def save_best_model_hparams(log_folder: str, average_df: pd.DataFrame) -> None:
    """Save best model hyperparameters to a JSON file.

    Args:
        log_folder (str): Path to the log folder.
        average_df: DataFrame containing averaged results.
    """
    best_model = find_best_hparams(average_df, mode="loss")
    best_hparams_path = f"{log_folder}/best_model_hparams.json"
    if not os.path.exists(best_hparams_path):
        save_best_hparams_df(best_hparams_path, best_model)


def main() -> None:
    """Main function to execute the script."""
    args = parse_arguments()
    hyperparam_dict = get_hyperparam_dict(args)
    process_results(args.log_dir, hyperparam_dict, args.learner)


if __name__ == "__main__":
    main()
