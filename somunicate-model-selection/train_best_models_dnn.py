# imports
import argparse
import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from DNN_models import DNNRegressor
from utils.gridsearch import get_stratified_train_test_split, get_target_tensor
from utils.utils import load_global_variables

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

try:
    hyperparams = pd.read_json(args.hparam_file, orient="records").to_dict(
        orient="records"
    )[0]
except Exception as exception:
    print(f"Could not read hyperparameter file {args.hparam_file}: {exception}")
    exit(1)


data_df = pd.read_json("./data/240918_median_data_3_features.json", orient="records")

# get subset from hyperparameters (we have to distinguish between onedimensional and all)
# 'all' has 'all' as hyperparameter target_name
# 'dimensions' has the target name as hyperparameter target_name
if hyperparams["target"] in global_variables["target_list"]:
    subset = "dimensions"
else:
    subset = hyperparams["target"]


if subset == "dimensions" or subset == "all":
    target_df = data_df[global_variables["target_list"]]
elif subset == "status":
    target_df = data_df[global_variables["status_list"]]
elif subset == "appeal":
    target_df = data_df[global_variables["appeal_list"]]
elif subset == "brand_identity":
    target_df = data_df[global_variables["brand_identity_list"]]


# for both 'all' and 'dimensions' the target_df has all targets
# for the onedimensional case the target tensor (1dim) has to be extracted differently
if subset == "dimensions":
    # find target index
    target_idx = global_variables["target_list"].index(hyperparams["target"])
    target_name = global_variables["target_list"][target_idx]
    target_tensor = get_target_tensor(target_df, target_idx)
    print(target_name, target_idx, target_tensor.shape)
else:
    target_tensor = get_target_tensor(target_df)
    target_name = subset
    print(target_name, target_tensor.shape)


result_folder = f"{args.log_folder}/{target_name}/"
os.makedirs(result_folder, exist_ok=True)
result_df = pd.DataFrame(
    columns=[
        "target",
        "folds",
        "experiment_id",
        "test_loss",
        "test_r2",
    ]
)


feature_df = data_df.drop(columns=global_variables["target_list"]).drop(
    columns=["sound"]
)
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
if len(train_indices) % 16 == 1 or len(train_indices) % 32 == 1:
    train_indices = train_indices[:-1]
    print("Removed one sample from train set")
if len(valid_indices) % 16 == 1 or len(valid_indices) % 32 == 1:
    valid_indices = valid_indices[:-1]
    print("Removed one sample from validation set")

test_indices_file = f"{result_folder}test_indices.json"
with open(test_indices_file, "w") as f:
    json.dump(test_indices.tolist(), f)

if len(test_indices) % 16 == 1 or len(test_indices) % 32 == 1:
    test_indices = test_indices[:-1]
    print("Removed one sample from test set")

print(
    "Train",
    len(train_indices),
    "Validation",
    len(valid_indices),
    "Test",
    len(test_indices),
)

train_set = torch.utils.data.Subset(train_val_dataset, train_indices)
valid_set = torch.utils.data.Subset(train_val_dataset, valid_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)
train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True)
valid_loader = DataLoader(
    valid_set, batch_size=hyperparams["batch_size"], shuffle=False
)
test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=False)

input_dim = train_loader.dataset[0][0].shape[0]
output_dim = train_loader.dataset[0][1].shape[0]

for e_idx in range(args.n_experiments):
    model = DNNRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hyperparams["layer_config"],
        use_batch_norm=hyperparams["batch_norm"],
        dropout_prob=hyperparams["dropout"],
        learning_rate=hyperparams["start_lr"],
        lr_scheduler=True,
        weight_decay=True,
        target_name=target_name,
        loss_type="rmse",
    )

    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, valid_loader)

    # evaluate on test set
    trainer.test(model, test_loader)

    test_loss = trainer.callback_metrics.get("test_loss", None).item()
    test_r2 = trainer.callback_metrics.get("test_r2", None).item()
    result_df.loc[e_idx] = [target_name, args.n_folds, e_idx, test_loss, test_r2]
    result_df.to_json(f"{result_folder}results.json", orient="records", indent=4)

    # save predictions for complete dataset
    if subset == "dimensions":
        pred_columns = ["sound", target_name]
    else:
        pred_columns = ["sound"] + list(target_df.columns)
    pred_df = pd.DataFrame(columns=pred_columns)
    pred_df["sound"] = data_df["sound"]

    complete_dataset = TensorDataset(feature_tensor, target_tensor)
    complete_loader = DataLoader(
        complete_dataset, batch_size=hyperparams["batch_size"], shuffle=False
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
        f"{result_folder}predictions_{e_idx}.json", orient="records", indent=4
    )
