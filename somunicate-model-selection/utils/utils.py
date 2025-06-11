import json
import os
from pathlib import Path

import pandas as pd


def load_global_variables() -> dict:
    dict_path = Path(__file__).parent / "global_variables.json"
    with open(dict_path) as file:
        global_variables = json.load(file)
    return global_variables


def load_df(log_folder: str, file_name: str):
    results_path = f"{log_folder}/{file_name}"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} does not exist")
    else:
        print("Loading results from", results_path)
        return pd.read_json(results_path, orient="records")
