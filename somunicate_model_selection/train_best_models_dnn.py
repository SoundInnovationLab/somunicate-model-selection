# imports
import argparse
import logging
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from dnn_models import DNNRegressor, LossType, ModelConfig
from torch.utils.data import DataLoader, TensorDataset
from utils.gridsearch import get_target_tensor
from utils.gridsearch_tensor import get_stratified_train_test_split
from utils.loading import load_global_variables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DIMENSIONS = "dimensions"
STATUS = "status"
APPEAL = "appeal"
BRAND_IDENTITY = "brand_identity"
ALL = "all"

START_LR = "start_lr"
BATCH_SIZE = "batch_size"
BATCH_NORM = "batch_norm"
DROPOUT = "dropout"
LAYER_CONFIG = "layer_config"

ORIENT_RECORDS = "records"
TARGET = "target"
TARGET_LIST = "target_list"
SOUND = "sound"


global_variables = load_global_variables()

parser = argparse.ArgumentParser(description="Train DNN with best hyperparameters")
parser.add_argument("--hparam_file", type=str, help="File with best hyperparameters")
parser.add_argument(
    "--log_folder",
    type=str,
    default="logs/best_models_dnn",
    help="Log folder for results.",
)
parser.add_argument(
    "--n_folds",
    type=int,
    default=5,
    help="Number of folds for k-fold cross-validation.",
)
parser.add_argument(
    "--n_experiments", type=int, default=1, help="Number of repeated experiments."
)
parser.add_argument(
    "--include_industry", type=str, default="True", help="Include industry as feature."
)


def load_hyperparameters(hparam_file: str) -> dict:
    """Load hyperparameters from a JSON file.

    Args:
        hparam_file (str): Path to the hyperparameter file.

    Returns:
        dict: Dictionary containing hyperparameters.

    Raises:
        ValueError: If the file cannot be loaded.
    """
    try:
        return pd.read_json(hparam_file, orient=ORIENT_RECORDS).to_dict(
            orient=ORIENT_RECORDS
        )[0]
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Error loading hyperparameters: {exc}")
        raise


def prepare_target_data(
    data_df: pd.DataFrame, hyperparams: dict, global_variables: dict
) -> tuple[str, torch.Tensor]:
    """Prepare target data based on hyperparameters.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        hyperparams (dict): Dictionary of hyperparameters.
        global_variables (dict): Global variables.

    Returns:
        tuple[str, torch.Tensor]: Target name and target tensor.
    """
    subset = (
        DIMENSIONS
        if hyperparams[TARGET] in global_variables[TARGET_LIST]
        else hyperparams[TARGET]
    )

    if subset == DIMENSIONS:
        target_idx = global_variables[TARGET_LIST].index(hyperparams[TARGET])
        target_name = global_variables[TARGET_LIST][target_idx]
        target_tensor = get_target_tensor(
            data_df[global_variables[TARGET_LIST]], target_idx
        )
    else:
        target_tensor = get_target_tensor(data_df[global_variables[f"{subset}_list"]])
        target_name = subset

    return target_name, target_tensor


def prepare_feature_data(
    data_df: pd.DataFrame, global_variables: dict, include_industry: str
) -> torch.Tensor:
    """Prepare feature data.

    Args:
        data_df (pd.DataFrame): Dataframe containing the dataset.
        global_variables (dict): Global variables.
        include_industry (str): Whether to include industry features.

    Returns:
        torch.Tensor: Feature tensor.
    """
    feature_df = data_df.drop(columns=global_variables[TARGET_LIST]).drop(
        columns=[SOUND]
    )
    if include_industry == "False":
        industry_columns = [col for col in feature_df.columns if "topic" not in col]
        feature_df = feature_df.drop(columns=industry_columns)
    return torch.tensor(feature_df.values, dtype=torch.float32)


def train_and_evaluate_model(  # noqa: WPS210
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    hyperparams: dict,
    dims: tuple[int, int],  # Tuple containing input_dim and output_dim
    result_folder: str,
    e_idx: int,
    n_folds: int,
) -> None:
    """Train and evaluate the model.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        hyperparams (dict): Dictionary of hyperparameters.
        dims (tuple[int, int]): Tuple containing input and output dimensions.
        result_folder (str): Folder to save results.
        e_idx (int): Experiment index.
        n_folds (int): Number of folds for cross-validation.
    """
    input_dim, output_dim = dims  # Unpack the tuple

    model_config = ModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        use_batch_norm=hyperparams[BATCH_NORM],
        dropout_prob=hyperparams[DROPOUT],
        learning_rate=hyperparams[START_LR],
        lr_scheduler=True,
        weight_decay=True,
        target_name=hyperparams["target"],
        loss_type=LossType.RMSE,
    )

    model = DNNRegressor(args=model_config, hidden_dims=hyperparams[LAYER_CONFIG])

    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    test_loss = trainer.callback_metrics.get("test_loss", None).item()
    test_r2 = trainer.callback_metrics.get("test_r2", None).item()

    result_df = pd.DataFrame(
        [[hyperparams["target"], n_folds, e_idx, test_loss, test_r2]],
        columns=[TARGET, "folds", "experiment_id", "test_loss", "test_r2"],
    )
    result_df.to_json(f"{result_folder}results.json", orient=ORIENT_RECORDS, indent=4)


def main() -> None:  # noqa: WPS210
    """Main function to train the best DNN model."""
    args = parser.parse_args()
    hyperparams = load_hyperparameters(args.hparam_file)
    data_df = pd.read_json("./data/dummy_audio_dataset.json", orient=ORIENT_RECORDS)
    global_variables = load_global_variables()

    target_name, target_tensor = prepare_target_data(
        data_df, hyperparams, global_variables
    )
    feature_tensor = prepare_feature_data(
        data_df, global_variables, args.include_industry
    )

    result_folder = f"{args.log_folder}/{target_name}/"
    os.makedirs(result_folder, exist_ok=True)

    dataset = TensorDataset(feature_tensor, target_tensor)
    train_indices, test_indices = get_stratified_train_test_split(
        dataset, n_folds=args.n_folds, non_parametric=True
    )
    train_val_dataset = TensorDataset(
        feature_tensor[train_indices], target_tensor[train_indices]
    )
    train_indices, valid_indices = get_stratified_train_test_split(
        train_val_dataset, n_folds=args.n_folds, non_parametric=True, train_val=True
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(train_val_dataset, train_indices),
        batch_size=hyperparams[BATCH_SIZE],
        shuffle=True,
    )
    valid_loader = DataLoader(
        torch.utils.data.Subset(train_val_dataset, valid_indices),
        batch_size=hyperparams[BATCH_SIZE],
        shuffle=False,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=hyperparams[BATCH_SIZE],
        shuffle=False,
    )

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]
    output_dim = sample_batch[1].shape[1]

    for e_idx in range(args.n_experiments):
        train_and_evaluate_model(
            train_loader,
            valid_loader,
            test_loader,
            hyperparams,
            (input_dim, output_dim),
            result_folder,
            e_idx,
            args.n_folds,
        )


if __name__ == "__main__":
    main()
