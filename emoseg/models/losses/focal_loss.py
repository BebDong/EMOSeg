import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.losses import weight_reduce_loss


@MODELS.register_module()
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2.0, loss_weight=1.0, reduction='mean', loss_name='loss_focal', **kwargs):
        super().__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.alpha = loss_weight
        self.reduction = reduction
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, target, ignore_index=255, **kwargs):
        """ using F.cross_entropy """
        # num_classes = pred.size(1)
        # target[target == ignore_index] = num_classes
        # onehot_target = F.one_hot(target.to(torch.int64), num_classes + 1).permute((0, 3, 1, 2)).to(torch.float)
        # onehot_target = onehot_target[:, :num_classes]  # filter ignored label
        # prob = F.softmax(pred, dim=1)
        # prob = torch.einsum('bc..., bc...->b...', onehot_target, prob)
        # weight = torch.pow(1.0 - prob, self.gamma)
        # loss = F.cross_entropy(pred, target, None, ignore_index=ignore_index, reduction='none')
        # loss = weight_reduce_loss(loss, weight=weight, reduction=self.reduction)
        # return loss * self.loss_weight

        """ https://github.com/Nacriema/Loss-Functions-For-Semantic-Segmentation/tree/master """
        num_classes = pred.size(1)
        target[target == ignore_index] = num_classes
        onehot_target = F.one_hot(target.to(torch.int64), num_classes + 1).permute((0, 3, 1, 2)).to(torch.float)
        onehot_target = onehot_target[:, :num_classes]  # filter ignored label
        prob = F.softmax(pred, dim=1)
        log_prob = F.log_softmax(pred, dim=1)
        weight = torch.pow(1.0 - prob, self.gamma)
        focal = -self.alpha * weight * log_prob
        loss = torch.einsum('bc..., bc...->b...', onehot_target, focal)
        loss = weight_reduce_loss(loss, reduction=self.reduction)
        return loss
