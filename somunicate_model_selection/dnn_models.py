from dataclasses import asdict, dataclass
from enum import Enum

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim  # noqa: WPS458
from utils.dnn_model import R2Score, RMSEMetric


class LossType(str, Enum):
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"


@dataclass
class ModelConfig:
    input_dim: int = 0  # Set a default value
    output_dim: int = 0  # Set a default value
    use_batch_norm: bool = False
    dropout_prob: float = 0.5
    learning_rate: float = 1e-3
    lr_scheduler: bool = True
    weight_decay: bool = True
    loss_type: LossType = LossType.RMSE
    target_name: str = ""


class BaseRegressor(pl.LightningModule):
    """
    Base class for regression models using PyTorch Lightning.
    """

    def __init__(self, args: ModelConfig) -> None:
        """
        Initializes the BaseRegressor model.

        Args:
            args (ModelConfig): An object containing all the arguments.
        """
        super().__init__()

        self._base_regressor_args = args  # Store args in a single attribute
        self._initialize_metrics()
        self.save_hyperparameters(asdict(args))

    def configure_optimizers(self):
        if self._base_regressor_args.weight_decay:
            optimizer = optim.Adam(
                self.parameters(),
                lr=self._base_regressor_args.learning_rate,
                weight_decay=1e-3,
            )
        else:
            optimizer = optim.Adam(
                self.parameters(), lr=self._base_regressor_args.learning_rate
            )

        if self._base_regressor_args.lr_scheduler:
            # values are fixed for experients
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=5, factor=0.5
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid_loss",  # the metric to be monitored
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.train_loss(predictions, targets)
        r2 = self.train_r2(predictions, targets)
        self.log_dict(
            {"train_loss": loss, "train_r2": r2}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.valid_loss(predictions, targets)
        r2 = self.valid_r2(predictions, targets)
        self.log_dict({"valid_loss": loss, "valid_r2": r2}, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.test_loss(predictions, targets)
        r2 = self.test_r2(predictions, targets)
        self.log_dict({"test_loss": loss, "test_r2": r2})
        return loss

    def _initialize_metrics(self) -> None:
        """
        Initializes the metrics for training, validation, and testing based on the specified loss type.
        """
        if self._base_regressor_args.loss_type == "mse":
            self.train_loss = torchmetrics.regression.MeanSquaredError()
            self.valid_loss = torchmetrics.regression.MeanSquaredError()
            self.test_loss = torchmetrics.regression.MeanSquaredError()
        elif self._base_regressor_args.loss_type == "mae":
            self.train_loss = torchmetrics.regression.MeanAbsoluteError()
            self.valid_loss = torchmetrics.regression.MeanAbsoluteError()
            self.test_loss = torchmetrics.regression.MeanAbsoluteError()
        elif self._base_regressor_args.loss_type == "rmse":
            self.train_loss = RMSEMetric()
            self.valid_loss = RMSEMetric()
            self.test_loss = RMSEMetric()
        else:
            raise ValueError("Invalid loss_type")

        self.train_r2 = R2Score()
        self.valid_r2 = R2Score()
        self.test_r2 = R2Score()


class DNNRegressor(BaseRegressor):
    """
    A deep neural network regressor that extends the BaseRegressor class.
    This model is designed to handle the regression tasks with customizable hidden layers.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        args: ModelConfig,
    ) -> None:
        super().__init__(args)

        layers = []
        current_dim = args.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if args.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if args.dropout_prob > 0:
                layers.append(nn.Dropout(args.dropout_prob))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, args.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            input_tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        output = input_tensor
        for layer in self.network:
            output = layer(output)
        return output
