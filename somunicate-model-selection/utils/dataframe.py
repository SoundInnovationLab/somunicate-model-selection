import numpy as np
import pandas as pd

VAL_LOSS_MEAN = "val_loss_mean"
VAL_R2_MEAN = "val_r2_mean"


def _append_best_model(best_df: pd.DataFrame, best_model: pd.DataFrame) -> pd.DataFrame:
    """Appends the best model to the existing DataFrame."""
    best_model_dict = best_model.iloc[0].to_dict()
    new_row = pd.DataFrame([best_model_dict], columns=best_df.columns)
    return pd.concat([best_df, new_row], ignore_index=True)


def _filter_dataframe(df, hyperparam):
    """Filters the DataFrame based on hyperparameter values."""
    filtered_df = df
    for key, hparam_value in hyperparam.items():
        filtered_df = _apply_filter(filtered_df, key, hparam_value)
    return filtered_df


def _apply_filter(df, key, hparam_value):
    """Applies a specific filter to the DataFrame based on the key."""
    if key == "layer_config":
        hparam_value = str(hparam_value)
        return df[df[key].astype(str) == hparam_value]
    elif key == "dropout":
        return df[np.isclose(df[key], hparam_value, atol=1e-9)]
    elif key == "max_features" or key == "max_depth":
        return df[df[key].isnull()]
    else:
        return df[df[key] == hparam_value]


def _get_hparam_columns(mode: str) -> list[str]:
    """Returns a list of hyperparameter column names based on the mode."""
    if mode == "dnn":
        return [
            "start_lr",
            "batch_size",
            "batch_norm",
            "dropout",
            "layer_config",
        ]
    elif mode == "rf":
        return [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _create_new_row(filtered_df, hparam_columns):
    """Creates a new row with hyperparameter values and mean loss/r2."""
    new_row = filtered_df[hparam_columns].iloc[0]
    new_row[VAL_LOSS_MEAN] = filtered_df["val_loss"].mean()
    new_row[VAL_R2_MEAN] = filtered_df["val_r2"].mean()
    new_row["target"] = filtered_df["target"].iloc[0]
    return new_row
