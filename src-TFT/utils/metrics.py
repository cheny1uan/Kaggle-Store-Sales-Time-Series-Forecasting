import torch
from pytorch_forecasting.metrics import MultiHorizonMetric


class RMSLELoss(MultiHorizonMetric):
    """PyTorch Forecasting 原生 RMSLE 损失。"""

    def __init__(self, reduction: str = "sqrt-mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.clamp(self.to_prediction(y_pred), min=0.0)
        target = torch.clamp(target, min=0.0)
        return torch.pow(torch.log1p(prediction) - torch.log1p(target), 2)

