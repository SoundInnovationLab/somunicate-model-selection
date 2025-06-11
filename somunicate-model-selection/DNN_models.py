import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from utils.dnn_model import R2Score, RMSEMetric


class BaseRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        use_batch_norm=False,
        dropout_prob=0.5,
        learning_rate=1e-3,
        lr_scheduler=False,
        weight_decay=False,
        loss_type="mse",
        target_name="",
        demographic_data=False,
        predict_fam_lik=False,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.target_name = target_name
        self.demographic_data = demographic_data
        self.predict_fam_lik = predict_fam_lik
        self._initialize_metrics()
        self.save_hyperparameters()

    def _initialize_metrics(self):
        if self.loss_type == "mse":
            self.train_loss = torchmetrics.regression.MeanSquaredError()
            self.valid_loss = torchmetrics.regression.MeanSquaredError()
            self.test_loss = torchmetrics.regression.MeanSquaredError()
        elif self.loss_type == "mae":
            self.train_loss = torchmetrics.regression.MeanAbsoluteError()
            self.valid_loss = torchmetrics.regression.MeanAbsoluteError()
            self.test_loss = torchmetrics.regression.MeanAbsoluteError()
        elif self.loss_type == "rmse":
            self.train_loss = RMSEMetric()
            self.valid_loss = RMSEMetric()
            self.test_loss = RMSEMetric()
        else:
            raise ValueError("Invalid loss_type")

        self.train_r2 = R2Score()
        self.valid_r2 = R2Score()
        self.test_r2 = R2Score()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.train_loss(y_hat, y)
        r2 = self.train_r2(y_hat, y)
        self.log_dict(
            {"train_loss": loss, "train_r2": r2}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.valid_loss(y_hat, y)
        r2 = self.valid_r2(y_hat, y)
        self.log_dict({"valid_loss": loss, "valid_r2": r2}, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.test_loss(y_hat, y)
        r2 = self.test_r2(y_hat, y)
        self.log_dict({"test_loss": loss, "test_r2": r2})
        return loss

    def configure_optimizers(self):
        if self.weight_decay:
            optimizer = optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=1e-3
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.lr_scheduler:
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


class DNNRegressor(BaseRegressor):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        use_batch_norm=False,
        dropout_prob=0.5,
        learning_rate=1e-3,
        lr_scheduler=False,
        weight_decay=False,
        loss_type="mse",
        target_name="",
        demographic_data=False,
        predict_fam_lik=False,
    ):
        super().__init__(
            input_dim,
            output_dim,
            use_batch_norm,
            dropout_prob,
            learning_rate,
            lr_scheduler,
            weight_decay,
            loss_type,
            target_name,
            demographic_data,
        )

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.network):
            x = layer(x)
        return x
