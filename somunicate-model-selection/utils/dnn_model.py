import torch
from sklearn.metrics import r2_score
from torchmetrics import Metric


class R2Score(Metric):
    """
    R2 Score metric for Regression Learner.
    """

    def __init__(self):
        super().__init__()
        self.add_state("y_true", default=torch.tensor([]), dist_reduce_fx=None)
        self.add_state("y_pred", default=torch.tensor([]), dist_reduce_fx=None)

    def update(self, y_pred, y_true):
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        self.y_true = self.y_true.cpu()
        self.y_pred = self.y_pred.cpu()
        self.y_true = torch.cat((self.y_true, y_true))
        self.y_pred = torch.cat((self.y_pred, y_pred))

    def compute(self):
        if len(self.y_true) == 0:
            return torch.tensor(float("nan"))

        # sklearn uses opposite order of arguments
        r2 = r2_score(self.y_true, self.y_pred, multioutput="variance_weighted")
        return torch.tensor(r2, dtype=torch.float32)


class RMSEMetric(Metric):
    """
    Root Mean Squared Error metric for Regression Learner.
    """

    def __init__(self):
        super().__init__()
        # Initialize accumulators for sum of squared errors and total number of samples
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
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
        if self.total_samples == 0:
            return torch.tensor(
                0.0
            )  # To handle division by zero if no samples were added

        mse = self.sum_squared_error / self.total_samples
        return torch.sqrt(mse)
