import json
import os
from pathlib import Path

import pandas as pd


def load_global_variables() -> dict:
    dict_path = Path(__file__).parent / "global_variables.json"
    with open(dict_path) as dict_file:
        global_variables = json.load(dict_file)
    return global_variables


def load_df(log_folder: str, file_name: str):
    results_path = f"{log_folder}/{file_name}"
    if os.path.exists(results_path):
        return pd.read_json(results_path, orient="records")
    else:
        raise FileNotFoundError(f"{results_path} does not exist")
