import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class ContextFlow(BaseModule):
    def __init__(self, scale, in_channels, channels, norm_cfg=None, act_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        self.scale = scale
        """ series-parallel """
        self.conv1 = DepthwiseSeparableConvModule(in_channels + channels, channels, 3, 1, 1,
                                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(channels, channels, 3, 1, 1,
                                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        """ parallel """
        # self.conv1 = DepthwiseSeparableConvModule(in_channels, channels, 3, 1, 1,
        #                                           norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.conv2 = DepthwiseSeparableConvModule(channels, channels, 3, 1, 1,
        #                                           norm_cfg=norm_cfg, act_cfg=act_cfg)

    """ series-parallel"""

    def forward(self, x_trunk, x_cf):
        h, w = x_trunk.shape[2:]
        out = torch.cat([x_trunk, x_cf], dim=1)
        out = F.adaptive_avg_pool2d(out, output_size=(int(h / self.scale), int(w / self.scale)))
        out = self.conv1(out)
        out = self.conv2(out)
        out = resize(out, size=(h, w), mode='bilinear', align_corners=False)
        return out

    """ parallel """
    # def forward(self, x_trunk):
    #     h, w = x_trunk.shape[2:]
    #     out = F.adaptive_avg_pool2d(x_trunk, output_size=(int(h / self.scale), int(w / self.scale)))
    #     out = self.conv1(out)
    #     out = self.conv2(out)
    #     out = resize(out, size=(h, w), mode='bilinear', align_corners=False)
    #     return out


class FSModule(BaseModule):
    def __init__(self, in_channels, channels, norm_cfg=None, act_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        self.conv3x3 = ConvModule(in_channels, channels, 3, 1, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1x1 = ConvModule(channels, channels, 1, norm_cfg=norm_cfg,
                                  act_cfg=dict(type='Sigmoid'))

    def forward(self, x):
        x = self.conv3x3(x)
        score = self.pool(x)
        score = self.conv1x1(score)
        out = x + score * x
        return out


class SPHModule(BaseModule):
    def __init__(self, scales, in_channels, channels, norm_cfg=None, act_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        self.gf = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            ConvModule(in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.branches = ModuleList()
        for scale in scales:
            self.branches.append(ContextFlow(scale, in_channels, channels, norm_cfg, act_cfg))
        self.fsm = FSModule(channels, channels, norm_cfg, act_cfg)

    def forward(self, x):
        gf_feat = resize(self.gf(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        outs = [gf_feat]
        for branch in self.branches:
            outs.append(branch(x, outs[-1]))  # series-parallel
            # outs.append(branch(x))  # parallel
        outs = [torch.unsqueeze(out, dim=0) for out in outs]
        outs = torch.cat(outs, dim=0)
        out = torch.sum(outs, dim=0, keepdim=False)
        out = self.fsm(out)
        return out


@MODELS.register_module()
class CAHeadV1(BaseDecodeHead):
    def __init__(self, scales=(2, 4, 8, 16), decoder_channels=48, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert isinstance(scales, (list, tuple))
        # self.decoder = ConvModule(self.in_channels[0], decoder_channels, 3, 1, 1,
        #                           norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        # self.conv3x3 = ConvModule(decoder_channels + self.channels, self.channels, 3, 1, 1,
        #                           norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.cam = SPHModule(scales, self.in_channels[1], self.channels, self.norm_cfg, self.act_cfg)
        self.conv3x3 = ConvModule(self.channels, self.channels, 3, 1, 1,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        """ w/ decoder """
        # low = self.decoder(x[0])
        # high = resize(self.cam(x[1]), size=low.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # out = torch.cat([low, high], dim=1)
        # out = self.conv3x3(out)
        # out = self.cls_seg(out)
        # return out
        """ w/o decoder """
        out = self.cam(x[1])
        out = self.conv3x3(out)
        out = self.cls_seg(out)
        return out
