import torch
from sklearn.metrics import r2_score
from torchmetrics import Metric


class R2Score(Metric):
    """
    Computes the R2 score for regression tasks.

    Attributes:
        y_true (torch.Tensor): Tensor to store true target values.
        y_pred (torch.Tensor): Tensor to store predicted values.
    """

    def __init__(self):
        super().__init__()
        self.add_state("y_true", default=torch.tensor([]), dist_reduce_fx=None)
        self.add_state("y_pred", default=torch.tensor([]), dist_reduce_fx=None)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Updates the state with new predictions and true values.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True target values.
        """
        y_true = y_true.detach().cpu().to(torch.float32)
        y_pred = y_pred.detach().cpu().to(torch.float32)
        self.y_true: torch.Tensor = self.y_true.cpu()
        self.y_pred: torch.Tensor = self.y_pred.cpu()
        self.y_true = torch.cat((self.y_true, y_true))
        self.y_pred = torch.cat((self.y_pred, y_pred))

    def compute(self):
        """
        Computes the R2 score.

        Returns:
            torch.Tensor: The computed R2 score.
        """
        if len(self.y_true) == 0:
            return torch.tensor(float("nan"))

        # sklearn uses opposite order of arguments
        r2 = r2_score(self.y_true, self.y_pred, multioutput="variance_weighted")
        return torch.tensor(r2, dtype=torch.float32)


class RMSEMetric(Metric):
    """
    Computes the Root Mean Squared Error (RMSE) for regression tasks.

    Attributes:
        sum_squared_error (torch.Tensor): Accumulator for the sum of squared errors.
        total_samples (torch.Tensor): Accumulator for the total number of samples.
    """

    def __init__(self):
        super().__init__()
        # Initialize accumulators for sum of squared errors and total number of samples
        self.add_state(
            "sum_squared_error",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Flatten tensors to ensure 1D
        preds = preds.view(-1)
        targets = targets.view(-1)
        squared_error = torch.sum((preds - targets) ** 2)
        self.sum_squared_error += squared_error
        self.total_samples += preds.size(0)

    def compute(self):
        """
        Computes the RMSE.

        Returns:
            torch.Tensor: The computed RMSE value.
        """
        if self.total_samples == 0:
            return torch.tensor(
                0, dtype=torch.float32
            )  # To handle division by zero if no samples were added

        mse = self.sum_squared_error / self.total_samples
        return torch.sqrt(mse)
