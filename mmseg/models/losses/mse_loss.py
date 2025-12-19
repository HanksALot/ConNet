import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred, target):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        dice_losses = binary_mse_loss(
            pred[:, i],
            target.float())
        total_loss += dice_losses
    return total_loss / num_classes


@weighted_loss
def binary_mse_loss(pred, target):
    """Wrapper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0,
                 loss_name='loss_mse'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # pred = pred
        # loss = self.loss_weight * mse_loss(
        #     pred, target, weight, reduction=reduction, avg_factor=avg_factor)

        pred = torch.sigmoid(pred)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)

        # pred = torch.softmax(pred, dim=1)
        # loss = self.loss_weight * mse_loss(
        #     pred, target, weight, reduction=reduction, avg_factor=avg_factor)

        # pred = pred.squeeze(1)
        # loss_sum = torch.nn.functional.smooth_l1_loss(pred, target, reduction='sum')  #
        # loss = loss_sum / target.sum()
        # loss = self.loss_weight * loss

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
