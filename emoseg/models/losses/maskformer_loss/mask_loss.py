import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .criterion import SetCriterion
from .criterion_point import SetCriterion_point


@MODELS.register_module()
class MaskLoss(nn.Module):
    def __init__(self, num_classes, mask_weight=20.0, dice_weight=1.0,
                 cls_weight=1.0, loss_weight=1.0, use_point=False):
        super(MaskLoss, self).__init__()
        self.ignore_index = 255
        weight_dict = {'loss_ce': cls_weight,
                       'loss_mask': mask_weight,
                       'loss_dice': dice_weight}
        if use_point:
            self.criterion = SetCriterion_point(
                num_classes,
                weight_dict=weight_dict,
                losses=['masks']
            )
        else:
            self.criterion = SetCriterion(
                num_classes,
                weight_dict=weight_dict,
                losses=['masks']
            )
        self.loss_weight = loss_weight

    def prepare_targets(self, targets):
        new_targets = []
        for target_per_image in targets:
            gt_cls = target_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(target_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(target_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append(
                {
                    'labels': gt_cls,
                    'masks': masks
                }
            )
        return new_targets

    def forward(self, outputs, label, ignore_index=255):
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                losses.pop(k)  # remove this loss if not specified in 'weight_dict'
        return losses
