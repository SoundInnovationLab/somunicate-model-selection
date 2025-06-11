import argparse
import itertools
import os
from pathlib import Path

from utils.evaluation import (
    average_over_hparam_combinations,
    find_best_hparams,
    save_best_hparams_df,
)
from utils.utils import load_df

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
    hyperparam_dict = {
        "start_lr": [0.001, 0.0001],
        "batch_size": [32],
        "batch_norm": [False],
        "dropout": [0.5],
        "layer_config": [
            [128, 256],
        ],
    }
elif args.learner == "rf":
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

# for random forest
hyperparam_combinations = list(itertools.product(*hyperparam_dict.values()))


log_dir = args.log_dir

# go through all subfolders and search for file called "results.json"
# when a gridsearch was done for every target dimension (with all numbers of folds)
# this gets tedious, so all subdirs are searched
for subdir, _, files in os.walk(log_dir):
    for file in files:
        if file == "results.json":
            log_folder = subdir
            df = load_df(subdir, file)
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
            print(best_model)
            if not os.path.exists(f"{log_folder}/best_model_hparams.json"):
                save_best_hparams_df(
                    f"{log_folder}/best_model_hparams.json", best_model
                )
