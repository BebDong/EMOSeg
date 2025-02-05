import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import cnn
from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS
from mmseg.models.utils import resize, Upsample
from mmseg.models.losses import accuracy
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class CFEmbedding(BaseModule):
    def __init__(self, embed_dims, num_tokens, num_groups=4, bias=True, init_cfg=None):
        super(CFEmbedding, self).__init__(init_cfg)
        self.learner = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1, groups=num_groups, bias=False),
            nn.Conv2d(embed_dims, num_tokens, 1, bias=False)
        )
        self.conv = nn.Conv2d(embed_dims, embed_dims, 1, groups=num_groups, bias=False)
        self.embed = nn.Linear(embed_dims, embed_dims * 2, bias=bias)

    def forward(self, x):
        outs = []
        mask = self.learner(x)
        outs.append(mask)

        B, L, _, _ = mask.shape
        mask = mask.reshape(B, L, -1)
        mask = F.softmax(mask, dim=-1)

        x = self.conv(x)
        B, C, _, _ = x.shape
        x = x.reshape(B, C, -1)

        out = mask @ x.transpose(-2, -1)  # BLC
        out = self.embed(out)
        outs.append(out.transpose(-2, -1))  # BCL
        return outs


class CFTransformation(BaseModule):  # a.k.a multi-head cross attention
    def __init__(self, embed_dims, num_heads, kv_tokens, attn_drop_rate=.0,
                 drop_rate=.0, qkv_bias=True, qk_scale=None, proj_bias=True,
                 init_cfg=None):
        super(CFTransformation, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims ** -0.5

        self.q = cnn.DepthwiseSeparableConvModule(embed_dims, embed_dims, 3, 1, 1,
                                                  act_cfg=None, bias=qkv_bias)
        self.kv = CFEmbedding(embed_dims, kv_tokens, num_heads, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, query, key_value):
        B, _, H, W = query.shape  # high resolution & low-level
        _, _, h, w = key_value.shape  # low resolution & high-level

        q = self.q(query)
        mask, kv = self.kv(key_value)
        k, v = torch.chunk(kv, chunks=2, dim=1)
        outs = {'mask': mask}

        q = q.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.max(attn, -1, keepdim=True)[0].expand_as(attn) - attn  # stable training
        attn = F.softmax(attn, dim=-1)  # B(num_heads)(HW)L
        outs.update({'attn': torch.mean(attn, dim=1, keepdim=False)})
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(-2, -1).reshape(B, self.embed_dims, H, W)
        out = self.proj_drop(self.proj(out))
        outs.update({'out': out})

        return outs


class CFTModule(BaseModule):
    def __init__(self, embed_dims=256, num_heads=4, kv_tokens=150, attn_drop_rate=.0,
                 drop_rate=.0, qkv_bias=True, mlp_ratio=4, norm_cfg=None, init_cfg=None):
        super(CFTModule, self).__init__(init_cfg)
        _, self.norm_low = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        _, self.norm_high = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        self.cross_attn = CFTransformation(embed_dims, num_heads, kv_tokens, attn_drop_rate,
                                           drop_rate, qkv_bias=qkv_bias)

        _, self.norm_mlp = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        ffn_channels = embed_dims * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dims, ffn_channels, 1, bias=True),
            nn.Conv2d(ffn_channels, ffn_channels, 3, 1, 1, groups=ffn_channels, bias=True),
            cnn.build_activation_layer(dict(type='GELU')),
            nn.Dropout(drop_rate),
            nn.Conv2d(ffn_channels, embed_dims, 1, bias=True),
            nn.Dropout(drop_rate)
        )

    def forward(self, low, high):
        query = self.norm_low(low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        key_value = self.norm_high(high.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outs = self.cross_attn(query, key_value)

        out = outs.pop('out') + low
        out = self.mlp(self.norm_mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) + out
        outs.update({'out': out})
        return outs


@MODELS.register_module()
class CFTHead(BaseDecodeHead):
    def __init__(self, fpn_up=False, feature_strides=(4, 8, 16, 32), num_heads=4, attn_drop_rate=.0,
                 drop_rate=.0, qkv_bias=True, mlp_ratio=4, ln_norm_cfg=None, **kwargs):
        super(CFTHead, self).__init__(input_transform='multiple_select', **kwargs)
        if fpn_up:
            assert feature_strides is not None, "Need feature strides to successively up-sampling."
            assert len(feature_strides) == len(self.in_channels), "Mismatching strides and input features."
        self.fpn_up = fpn_up
        self.feature_strides = feature_strides

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.cfts = ModuleList()
        for i, in_channel in enumerate(self.in_channels):
            self.lateral_convs.append(nn.Conv2d(in_channel, self.channels, 1, bias=True))
            if i < len(self.in_channels) - 1:
                self.cfts.append(CFTModule(self.channels, num_heads, self.num_classes, attn_drop_rate,
                                           drop_rate, qkv_bias, mlp_ratio, norm_cfg=ln_norm_cfg))
                self.fpn_convs.append(nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True))
                # self.fpn_convs.append(nn.Identity())

        if fpn_up:
            self.scale_heads = nn.ModuleList()
            for i in range(len(feature_strides)):
                head_length = max(1, int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
                scale_head = []
                for k in range(head_length):
                    scale_head.append(cnn.ConvModule(self.channels, self.channels, 3, 1, 1,
                                                     norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                    if feature_strides[i] != feature_strides[0]:
                        scale_head.append(Upsample(scale_factor=2, mode='bilinear',
                                                   align_corners=self.align_corners))
                self.scale_heads.append(nn.Sequential(*scale_head))
        else:
            self.scale_heads = cnn.ConvModule(len(self.in_channels) * self.channels, self.channels,
                                              3, 1, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        high = laterals[-1]
        masks = []
        attns = []
        fpn_outs = [high]
        for i in range(len(self.cfts), 0, -1):
            low = laterals[i - 1]
            cft_outs = self.cfts[i - 1](low, high)
            high = cft_outs['out']
            masks.append(cft_outs['mask'])
            attns.append(cft_outs['attn'])
            fpn_outs.append(self.fpn_convs[i - 1](high))

        if self.fpn_up:
            fpn_outs.reverse()
            out = self.scale_heads[0](fpn_outs[0])
            for i in range(1, len(self.feature_strides)):
                out = out + resize(self.scale_heads[i](fpn_outs[i]),
                                   size=out.shape[2:],
                                   mode='bilinear',
                                   align_corners=self.align_corners)
        else:
            h, w = fpn_outs[-1].shape[2:]
            fpn_outs = [resize(fpn_out, size=(h, w), mode='bilinear',
                               align_corners=self.align_corners) for fpn_out in fpn_outs]
            out = torch.cat(fpn_outs, dim=1)
            out = self.scale_heads(out)

        # return out
        out = self.cls_seg(out)
        return masks, out

    def predict(self, inputs, batch_img_metas, test_cfg):
        _, seg_logit = self.forward(inputs)
        return self.predict_by_feat(seg_logit, batch_img_metas)

    def loss(self, inputs, batch_data_samples, train_cfg):
        masks, seg_logit = self.forward(inputs)
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logit = resize(seg_logit, seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        masks = [resize(mask, seg_label.shape[2:], mode='bilinear',
                        align_corners=self.align_corners) for mask in masks]
        mask = torch.sum(torch.stack(masks, dim=0), dim=0, keepdim=False)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        loss_decode_ce = self.loss_decode[0]
        loss_decode_mask = self.loss_decode[1]

        loss['loss_ce'] = loss_decode_ce(seg_logit, seg_label, weight=seg_weight,
                                         ignore_index=self.ignore_index)
        loss.update(loss_decode_mask(dict(pred_masks=mask), seg_label,
                                     ignore_index=self.ignore_index))
        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss
