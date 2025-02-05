import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss, get_class_weight

import os
import logging


# logger = logging.getLogger('cyun')
# logger.setLevel(level=logging.DEBUG)
# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# handler = logging.FileHandler(os.path.join(root_dir, 'sensitivity.log'), encoding='UTF-8')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)


@torch.no_grad()
def sensitivity(pred, gamma=1.0, scale=False):
    n_class = pred.size(1)
    eps = torch.finfo(pred.dtype).eps
    score = F.softmax(pred, dim=1).clamp(min=eps, max=1. - eps)
    min_real = torch.finfo(pred.dtype).min
    log_score = F.log_softmax(pred, dim=1).clamp(min=min_real)
    entropy = -torch.sum(score * log_score, dim=1)
    sen = torch.pow(entropy / math.log(n_class), exponent=gamma)
    # logger.info(round(sen.mean().cpu().item(), 5))  # for plot
    if scale:
        sen = sen / (sen.mean() + eps)  # approximate the lr policy
    return sen


@MODELS.register_module()
class SensitiveLoss(nn.Module):
    def __init__(self, gamma=1.0, use_scale=False, reduction='mean', class_weight=None,
                 loss_weight=1.0, loss_name='loss_sensitive', **kwargs):
        super(SensitiveLoss, self).__init__()
        self.gamma = gamma
        self.use_scale = use_scale
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, target, reduction_override=None, scale_override=None, ignore_index=255, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        scale = scale_override if scale_override is not None else self.use_scale
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        sen = sensitivity(pred, gamma=self.gamma, scale=scale)
        loss = F.cross_entropy(pred, target, weight=class_weight, ignore_index=ignore_index, reduction='none')
        loss = weight_reduce_loss(loss, weight=sen, reduction=reduction)
        return loss * self.loss_weight
