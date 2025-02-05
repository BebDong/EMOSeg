import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class Bottleneck(BaseModule):
    def __init__(self, in_channels, channels, norm_cfg=None, act_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        inner_channels = 128
        self.conv1 = ConvModule(in_channels, inner_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(inner_channels, inner_channels, 3, 2, 1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(inner_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv4 = ConvModule(in_channels, channels, 1, 2, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(residual)
        return x


class EncoderDecoder(BaseModule):
    def __init__(self, scale, channels, norm_cfg=None, act_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        num_blocks = int(math.log(scale, 2))
        self.encoder = nn.Sequential()
        for _ in range(num_blocks):
            self.encoder.append(Bottleneck(channels, channels, norm_cfg, act_cfg))
        self.conv3x3 = ConvModule(channels, channels, 3, 1, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1x1 = ConvModule(channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        out = self.encoder(x)
        out = self.conv3x3(out)
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.conv1x1(out)
        return out


class CAModule(BaseModule):
    def __init__(self, scales=(2, 4, 8), in_channels=(1024, 2048), channels=512, norm_cfg=None,
                 act_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        num_branches = len(scales)
        # compression
        self.comp_c3 = ConvModule(in_channels[0], channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.comp_c4 = ConvModule(in_channels[1], channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # context perception
        self.branches = ModuleList()
        for scale in scales:
            self.branches.append(EncoderDecoder(scale, channels, norm_cfg=norm_cfg, act_cfg=act_cfg))
        # context aggregation
        self.gate = nn.Sequential(
            ConvModule(channels * (num_branches + 1), channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, num_branches + 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

    def forward(self, c3, c4):
        c3 = self.comp_c3(c3)
        c4 = self.comp_c4(c4)

        outs = [c3]
        for branch in self.branches:
            # outs.append(branch(c4))
            if len(outs) == 1:
                outs.append(branch(c4))
            else:
                outs.append(branch(c4 + outs[-1]))

        score = torch.cat(outs, dim=1)
        score = self.gate(score)
        score = F.softmax(score, dim=1)
        score = torch.unsqueeze(score, dim=2)  # BN1HW

        outs = [torch.unsqueeze(out, dim=1) for out in outs]
        out = torch.cat(outs, dim=1)  # BNCHW
        out = torch.sum(out * score, dim=1, keepdim=False)

        return out


@MODELS.register_module()
class CAHeadV2(BaseDecodeHead):
    def __init__(self, scales=(2, 4, 8), gap_channels=48, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert isinstance(scales, (list, tuple))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            ConvModule(self.in_channels[1], gap_channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        )
        self.cam = CAModule(scales, self.in_channels, self.channels, self.norm_cfg, self.act_cfg)
        self.conv3x3 = nn.Conv2d(self.channels + gap_channels, self.channels, kernel_size=1)

    # def cls_seg(self, feat):
    #     """Classify each pixel."""
    #     output = self.conv_seg(feat)
    #     if self.dropout is not None:
    #         output = self.dropout(output)
    #     return output

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        out = self.cam(x[0], x[1])
        gap = resize(self.gap(x[1]), size=out.shape[2:], mode='bilinear', align_corners=self.align_corners)
        out = torch.cat((out, gap), dim=1)
        out = self.conv3x3(out)
        out = self.cls_seg(out)
        return out
