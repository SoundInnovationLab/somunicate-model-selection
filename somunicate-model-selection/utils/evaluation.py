import itertools
import os
from typing import Literal

import pandas as pd
from somunicate_model_selection.utils.dataframe import (
    _append_best_model,
    _create_new_row,
    _filter_dataframe,
    _get_hparam_columns,
)

VAL_LOSS_MEAN = "val_loss_mean"
VAL_R2_MEAN = "val_r2_mean"


def average_over_hparam_combinations(df, hyperparam_dict, mode):  # noqa: WPS210
    """Averages performance metrics over hyperparameter combinations."""
    hyperparam_combinations = list(itertools.product(*hyperparam_dict.values()))
    columns = list(df.columns) + [VAL_LOSS_MEAN, VAL_R2_MEAN]
    average_df = pd.DataFrame(columns=columns).drop(columns=["model_version", "fold"])

    averaged_data = []
    for combination in hyperparam_combinations:
        hyperparam = dict(zip(hyperparam_dict.keys(), combination, strict=False))
        filtered_df = _filter_dataframe(df, hyperparam)
        hparam_columns = _get_hparam_columns(mode)
        new_row = _create_new_row(filtered_df, hparam_columns)
        averaged_data.append(new_row)

    average_df = pd.DataFrame(averaged_data)
    average_df = average_df.drop(columns=["val_loss", "val_r2"])
    return average_df


def find_best_hparams(
    average_df: pd.DataFrame, mode: Literal["loss", "r2"] = "loss"
) -> pd.DataFrame:
    if mode == "r2":
        # find maximum r2
        max_r2 = average_df[VAL_R2_MEAN].max()
        return average_df[average_df[VAL_R2_MEAN] == max_r2]
    else:
        # find minimum loss
        min_loss = average_df[VAL_LOSS_MEAN].min()
        return average_df[average_df[VAL_LOSS_MEAN] == min_loss]


def save_best_hparams_df(file_name: str, best_model: pd.DataFrame) -> None:
    """Saves the DataFrame containing the best hyperparameter combination to a JSON file."""
    if os.path.exists(file_name):
        best_df = pd.read_json(file_name, orient="records")
        best_df = _append_best_model(best_df, best_model)
    else:
        best_df = pd.DataFrame(columns=best_model.columns)
        best_df.loc[0] = list(best_model.iloc[0])
    best_df.to_json(file_name, orient="records", indent=4)
