import itertools
import os
from typing import Literal

import pandas as pd

from utils.dataframe import (
    append_best_model,
    create_new_row,
    filter_dataframe,
    get_hparam_columns,
)

VAL_LOSS_MEAN = "val_loss_mean"
VAL_R2_MEAN = "val_r2_mean"


def average_over_hparam_combinations(  # noqa: WPS210
    df: pd.DataFrame, hyperparam_dict: dict[str, list], mode: str
) -> pd.DataFrame:
    """
    Averages performance metrics over hyperparameter combinations.

    Args:
        df (pd.DataFrame): DataFrame containing performance metrics.
        hyperparam_dict (dict): Dictionary of hyperparameter combinations.
        mode (str): Mode of the learner ('dnn' or 'rf').

    Returns:
        pd.DataFrame: DataFrame with averaged performance metrics.
    """
    hyperparam_combinations = list(itertools.product(*hyperparam_dict.values()))
    columns = list(df.columns) + [VAL_LOSS_MEAN, VAL_R2_MEAN]
    average_df = pd.DataFrame(columns=columns).drop(columns=["model_version", "fold"])

    averaged_data = []
    for combination in hyperparam_combinations:
        hyperparam = dict(zip(hyperparam_dict.keys(), combination, strict=False))
        filtered_df = filter_dataframe(df, hyperparam)
        hparam_columns = get_hparam_columns(mode)
        new_row = create_new_row(filtered_df, hparam_columns)
        averaged_data.append(new_row)

    average_df = pd.DataFrame(averaged_data)
    return average_df


def find_best_hparams(
    average_df: pd.DataFrame, mode: Literal["loss", "r2"] = "loss"
) -> pd.DataFrame:
    """
    Finds the best hyperparameter combination based on the specified mode.

    Args:
        average_df (pd.DataFrame): DataFrame with averaged performance metrics.
        mode (Literal["loss", "r2"], optional): Metric to optimize ('loss' or 'r2'). Defaults to 'loss'.

    Returns:
        pd.DataFrame: DataFrame containing the best hyperparameter combination.
    """
    if mode == "r2":
        # find maximum r2
        max_r2 = average_df[VAL_R2_MEAN].max()
        return average_df[average_df[VAL_R2_MEAN] == max_r2]
    else:
        # find minimum loss
        min_loss = average_df[VAL_LOSS_MEAN].min()
        return average_df[average_df[VAL_LOSS_MEAN] == min_loss]


def save_best_hparams_df(file_name: str, best_model: pd.DataFrame) -> None:
    """
    Saves the DataFrame containing the best hyperparameter combination to a JSON file.

    Args:
        file_name (str): Path to the JSON file.
        best_model (pd.DataFrame): DataFrame containing the best hyperparameter combination.
    """
    if os.path.exists(file_name):
        best_df = pd.read_json(file_name, orient="records")
        best_df = append_best_model(best_df, best_model)
    else:
        best_df = pd.DataFrame(columns=best_model.columns)
        best_df.loc[0] = list(best_model.iloc[0])
    best_df.to_json(file_name, orient="records", indent=4)
