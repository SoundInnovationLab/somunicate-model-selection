# imports
import argparse
import json
import logging
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from dnn_models import DNNRegressor, LossType, ModelConfig
from torch.utils.data import DataLoader, TensorDataset
from utils.gridsearch import get_stratified_train_test_split, get_target_tensor
from utils.utils import load_global_variables

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

args = parser.parse_args()

hparam_file = args.hparam_file
try:
    hyperparams = pd.read_json(hparam_file, orient=ORIENT_RECORDS).to_dict(
        orient=ORIENT_RECORDS
    )[0]
except (ValueError, FileNotFoundError) as exc:
    logger.error(f"Error loading hyperparameters: {exc}")


data_df = pd.read_json("./data/dummy_audio_dataset.json", orient=ORIENT_RECORDS)

# get subset from hyperparameters (we have to distinguish between onedimensional and all)
# 'all' has 'all' as hyperparameter target_name
# 'dimensions' has the target name as hyperparameter target_name
if hyperparams[TARGET] in global_variables[TARGET_LIST]:
    subset = DIMENSIONS
else:
    subset = hyperparams[TARGET]


if subset == DIMENSIONS or subset == "all":
    target_df = data_df[global_variables[TARGET_LIST]]
elif subset == "status":
    target_df = data_df[global_variables["status_list"]]
elif subset == "appeal":
    target_df = data_df[global_variables["appeal_list"]]
elif subset == "brand_identity":
    target_df = data_df[global_variables["brand_identity_list"]]


# for both 'all' and 'dimensions' the target_df has all targets
# for the onedimensional case the target tensor (1dim) has to be extracted differently
if subset == DIMENSIONS:
    # find target index
    target_idx = global_variables[TARGET_LIST].index(hyperparams[TARGET])
    target_name = global_variables[TARGET_LIST][target_idx]
    target_tensor = get_target_tensor(target_df, target_idx)
else:
    target_tensor = get_target_tensor(target_df)
    target_name = subset


result_folder = f"{args.log_folder}/{target_name}/"
os.makedirs(result_folder, exist_ok=True)
result_df = pd.DataFrame(
    columns=[
        TARGET,
        "folds",
        "experiment_id",
        "test_loss",
        "test_r2",
    ]
)


feature_df = data_df.drop(columns=global_variables[TARGET_LIST]).drop(columns=[SOUND])
if args.include_industry == "False":
    industry_columns = [col for col in feature_df.columns if "topic" not in col]
    feature_df = feature_df.drop(columns=industry_columns)
feature_tensor = torch.tensor(feature_df.values, dtype=torch.float32)
dataset = TensorDataset(feature_tensor, target_tensor)

# get holdout (test) set
train_indices, test_indices = get_stratified_train_test_split(
    dataset, n_folds=args.n_folds, non_parametric=True
)
train_val_dataset = TensorDataset(
    feature_tensor[train_indices], target_tensor[train_indices]
)

train_indices, valid_indices = get_stratified_train_test_split(
    train_val_dataset, n_folds=args.n_folds, non_parametric=True, train_val=True
)
batch_sizes = [16, 32]
needs_trimming = any(len(train_indices) % bs == 1 for bs in batch_sizes)
if needs_trimming:
    train_indices = train_indices[:-1]

needs_trimming = any(len(valid_indices) % bs == 1 for bs in batch_sizes)
if needs_trimming:
    valid_indices = valid_indices[:-1]

test_indices_file = f"{result_folder}test_indices.json"
with open(test_indices_file, "w") as test_indice_file:
    json.dump(test_indices.tolist(), test_indice_file)

needs_trimming = any(len(test_indices) % bs == 1 for bs in batch_sizes)
if needs_trimming:
    test_indices = test_indices[:-1]

train_set = torch.utils.data.Subset(train_val_dataset, train_indices)
valid_set = torch.utils.data.Subset(train_val_dataset, valid_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)
train_loader = DataLoader(train_set, batch_size=hyperparams[BATCH_SIZE], shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=hyperparams[BATCH_SIZE], shuffle=False)
test_loader = DataLoader(test_set, batch_size=hyperparams[BATCH_SIZE], shuffle=False)

sample_batch = next(iter(train_loader))
input_tensor, target_tensor = sample_batch[0], sample_batch[1]

input_dim = input_tensor.shape[1]
output_dim = target_tensor.shape[1]

for e_idx in range(args.n_experiments):
    model_config = ModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        use_batch_norm=hyperparams[BATCH_NORM],
        dropout_prob=hyperparams[DROPOUT],
        learning_rate=hyperparams[START_LR],
        lr_scheduler=True,
        weight_decay=True,
        target_name=target_name,
        loss_type=LossType.RMSE,
    )

    model = DNNRegressor(
        args=model_config,
        hidden_dims=hyperparams[LAYER_CONFIG],
    )

    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, valid_loader)

    # evaluate on test set
    trainer.test(model, test_loader)

    test_loss = trainer.callback_metrics.get("test_loss", None).item()
    test_r2 = trainer.callback_metrics.get("test_r2", None).item()
    result_df.loc[e_idx] = [target_name, args.n_folds, e_idx, test_loss, test_r2]
    result_df.to_json(f"{result_folder}results.json", orient=ORIENT_RECORDS, indent=4)

    # save predictions for complete dataset
    if subset == DIMENSIONS:
        pred_columns = [SOUND, target_name]
    else:
        pred_columns = [SOUND] + list(target_df.columns)
    pred_df = pd.DataFrame(columns=pred_columns)
    pred_df[SOUND] = data_df[SOUND]

    complete_dataset = TensorDataset(feature_tensor, target_tensor)
    complete_loader = DataLoader(
        complete_dataset, batch_size=hyperparams[BATCH_SIZE], shuffle=False
    )
    model.eval()
    predictions = []

    for batch in complete_loader:
        features, _ = batch
        with torch.no_grad():
            pred = model(features)
            predictions.append(pred)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    pred_df[pred_columns[1:]] = predictions

    pred_df.to_json(
        f"{result_folder}predictions_{e_idx}.json", orient=ORIENT_RECORDS, indent=4
    )
