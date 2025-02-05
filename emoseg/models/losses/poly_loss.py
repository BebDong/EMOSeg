import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.losses import weight_reduce_loss


@MODELS.register_module()
class PolyLoss(nn.Module):
    """ PolyLoss-1: arXiv:2204.12511."""

    def __init__(self, epsilon=2.0, loss_weight=1.0, reduction='mean', loss_name='loss_poly', **kwargs):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.loss_weight = loss_weight
        self.reduction = reduction
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, target, ignore_index=255, **kwargs):
        num_classes = pred.size(1)
        target[target == ignore_index] = num_classes
        onehot_target = F.one_hot(target.to(torch.int64), num_classes + 1).permute((0, 3, 1, 2)).to(torch.float)
        onehot_target = onehot_target[:, :num_classes]  # filter ignored label
        prob = F.softmax(pred, dim=1)
        prob = torch.einsum('bc..., bc...->b...', onehot_target, prob)  # pt
        loss = F.cross_entropy(pred, target, reduction='none', ignore_index=ignore_index)
        loss = loss + self.epsilon * (1.0 - prob)
        loss = weight_reduce_loss(loss, reduction=self.reduction)
        return self.loss_weight * loss
