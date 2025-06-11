import itertools
import os

import numpy as np
import pandas as pd


def average_over_hparam_combinations(df, hyperparam_dict, mode):
    hyperparam_combinations = list(itertools.product(*hyperparam_dict.values()))

    # prepare averade df
    columns = list(df.columns) + ["val_loss_mean", "val_r2_mean"]
    average_df = pd.DataFrame(columns=columns).drop(columns=["model_version", "fold"])

    # go through all hyperparam combinations
    for c_idx, combination in enumerate(hyperparam_combinations):
        hyperparam = dict(zip(hyperparam_dict.keys(), combination))
        filtered_df = df
        for key, value in hyperparam.items():
            if key == "layer_config":
                # here the value is a list of integers
                # we need to convert it to a string
                value = str(value)
                filtered_df = filtered_df[filtered_df[key].astype(str) == value]
            elif key == "dropout":
                # somehow the 0.3 dropout has a small numerical error
                filtered_df = filtered_df[
                    np.isclose(filtered_df[key], value, atol=1e-9)
                ]
            elif key == "max_features" or key == "max_depth":
                # here the value is None, null or Nan
                filtered_df = filtered_df[filtered_df[key].isnull()]

            else:
                filtered_df = filtered_df[filtered_df[key] == value]

        # Create a new row with hyperparameter values
        if mode == "dnn":
            hparam_columns = [
                "start_lr",
                "batch_size",
                "batch_norm",
                "dropout",
                "layer_config",
            ]
        elif mode == "rf":
            hparam_columns = [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
            ]

        new_row = filtered_df[hparam_columns].iloc[0]
        new_row["val_loss_mean"] = filtered_df["val_loss"].mean()
        new_row["val_r2_mean"] = filtered_df["val_r2"].mean()
        new_row["target"] = filtered_df["target"].iloc[0]
        average_df.loc[c_idx] = new_row
    average_df = average_df.drop(columns=["val_loss", "val_r2"])

    return average_df


def find_best_hparams(average_df, mode="loss"):
    assert mode in ["loss", "r2"], 'mode must be either "loss" or "r2"'
    if mode == "r2":
        # find maximum r2
        max_r2 = average_df["val_r2_mean"].max()
        return average_df[average_df["val_r2_mean"] == max_r2]
    else:
        # find minimum loss
        min_loss = average_df["val_loss_mean"].min()
        return average_df[average_df["val_loss_mean"] == min_loss]


def save_best_hparams_df(file_name, best_model):
    # if it doesn't exist, create the file log_folder + 'best_model_hparams.json'
    # and save the best model with the first column "target_name"
    # if it exists, load and append the best model to the file
    if not os.path.exists(file_name):
        best_df = pd.DataFrame(columns=best_model.columns)
        best_df.loc[0] = list(best_model.iloc[0])
    else:
        best_df = pd.read_json(file_name, orient="records")
        new_row = pd.DataFrame([list(best_model.iloc[0])], columns=best_df.columns)
        best_df = pd.concat([best_df, new_row], ignore_index=True)
    # save best_df
    best_df.to_json(file_name, orient="records", indent=4)
