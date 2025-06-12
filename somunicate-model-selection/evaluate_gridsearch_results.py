import argparse
import itertools
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

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


N_ESTIMATORS = "n_estimators"
MAX_DEPTH = "max_depth"
MIN_SAMPLES_SPLIT = "min_samples_split"
MIN_SAMPLES_LEAF = "min_samples_leaf"
MAX_FEATURES = "max_features"
START_LR = "start_lr"
BATCH_SIZE = "batch_size"
BATCH_NORM = "batch_norm"
DROPOUT = "dropout"
LAYER_CONFIG = "layer_config"

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
    help="Deep Neural Network (dnn),  Random Forest (rf)",
)
args = parser.parse_args()

if args.learner == "dnn":
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
    hyperparam_grid = DNNHyperparameterGrid()
    hyperparam_dict = {
        START_LR: hyperparam_grid.start_lr,
        BATCH_SIZE: hyperparam_grid.batch_size,
        BATCH_NORM: hyperparam_grid.batch_norm,
        DROPOUT: hyperparam_grid.dropout,
        LAYER_CONFIG: hyperparam_grid.layer_config,
    }
elif args.learner == "rf":
    # hyperparam_dict = {
    #     "n_estimators": [100, 200, 300],
    #     "max_depth": [None, 10, 20],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    #     "max_features": [None, "sqrt", "log2"],
    # }
    hyperparameter_grid = RFHyperparameterGrid()
    hyperparam_dict = {
        N_ESTIMATORS: hyperparameter_grid.n_estimators,
        MAX_DEPTH: hyperparameter_grid.max_depth,
        MIN_SAMPLES_SPLIT: hyperparameter_grid.min_samples_split,
        MIN_SAMPLES_LEAF: hyperparameter_grid.min_samples_leaf,
        MAX_FEATURES: hyperparameter_grid.max_features,
    }

hyperparam_combinations = list(
    itertools.product(
        *[hval for hval in hyperparam_dict.values() if isinstance(hval, Iterable)]
    )
)
log_dir = args.log_dir

# go through all subfolders and search for file called "results.json"
# when a gridsearch was done for every target dimension (with all numbers of folds)
# this gets tedious, so all subdirs are searched
for subdir, _, files in os.walk(log_dir):
    for result_file in files:
        if result_file == "results.json":
            log_folder = subdir
            df = load_df(subdir, result_file)
            average_df = average_over_hparam_combinations(
                df, hyperparam_dict, args.learner
            )
            if not os.path.exists(f"{log_folder}/fold_average_results.json"):
                average_df.to_json(
                    f"{log_folder}/fold_average_results.json",
                    orient="records",
                    indent=4,
                )
            best_model = find_best_hparams(average_df, mode="loss")
            if not os.path.exists(f"{log_folder}/best_model_hparams.json"):
                save_best_hparams_df(
                    f"{log_folder}/best_model_hparams.json", best_model
                )


def main():
    """
    Evaluates grid search results by averaging performance metrics over hyperparameter combinations
    and identifying the best hyperparameter configuration.

    This script processes all subdirectories in the specified log directory, computes averages,
    and saves the results to JSON files.
    """
