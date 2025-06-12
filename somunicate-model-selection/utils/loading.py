import json
import os
from pathlib import Path

import pandas as pd


def load_global_variables() -> dict:
    """Loads global variables from a JSON file.

    Returns:
        dict: A dictionary containing global variables loaded from 'global_variables.json'.
    """
    dict_path = Path(__file__).parent / "global_variables.json"
    with open(dict_path) as dict_file:
        global_variables = json.load(dict_file)
    return global_variables


def load_df(log_folder: str, file_name: str):
    """Loads a DataFrame from a JSON file.

    Args:
        log_folder (str): The folder where the log file is located.
        file_name (str): The name of the JSON file to load.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the JSON file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    results_path = f"{log_folder}/{file_name}"
    if os.path.exists(results_path):
        return pd.read_json(results_path, orient="records")
    else:
        raise FileNotFoundError(f"{results_path} does not exist")
