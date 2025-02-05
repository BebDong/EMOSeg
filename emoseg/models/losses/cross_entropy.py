# Adapted from: https://github.com/lovelyyoshino/RHACrackNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss


def weighted_binary_cross_entropy(pred, label, pos_weight, weight=None, from_logit=False, reduction='mean',
                                  avg_factor=None, avg_non_ignore=False, ignore_index=255, epsilon=1e-7):
    # should mask out the ignored elements
    valid_mask = ((label >= 0) & (label != ignore_index)).float()
    if weight is not None:
        weight = weight * valid_mask
    else:
        weight = valid_mask
    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()

    # calculate loss
    if not from_logit:
        pred = torch.clamp(pred, min=epsilon, max=(1 - epsilon))
        pred = torch.log(pred / (1 - pred))
    max_val = torch.clamp(-1 * pred, min=0)
    balanced_weight = 1 + label * (pos_weight - 1)
    loss = (1 - label) * pred + balanced_weight * (
            torch.log(torch.exp(-1 * max_val) + torch.exp(-1 * pred - max_val)) + max_val)
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_cnt, neg_cnt, pos_weight_factor, loss_weight=1.0, reduction='mean',
                 from_logit=False, avg_non_ignore=False, loss_name='loss_wbce', **kwargs):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = (pos_cnt + neg_cnt) / (pos_weight_factor * pos_cnt)  # from paper
        # self.pos_weight = (pos_cnt + neg_cnt) / pos_cnt * pos_weight_factor  # from code
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.from_logit = from_logit
        self.avg_non_ignore = avg_non_ignore
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, target, weight=None, avg_factor=None, ignore_index=255, **kwargs):
        pred = F.sigmoid(pred).squeeze(1)  # sigmoid and B1HW --> BHW
        loss = weighted_binary_cross_entropy(pred, target, pos_weight=self.pos_weight,
                                             weight=weight, from_logit=self.from_logit,
                                             reduction=self.reduction, avg_factor=avg_factor,
                                             avg_non_ignore=self.avg_non_ignore, ignore_index=ignore_index)
        return self.loss_weight * loss
