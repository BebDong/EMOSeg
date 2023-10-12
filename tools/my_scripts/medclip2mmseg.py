import argparse
import os.path as osp
from collections import OrderedDict
import torch
from mmengine import mkdir_or_exist
from mmengine.runner import CheckpointLoader


def parse_args():
    default_ckpt_dir = '/cluster/work/cvl/qutang/pretrained'
    parser = argparse.ArgumentParser()
    parser.add_argument('-src',
                        # default=osp.join(default_ckpt_dir, 'medclip_swin_tiny.bin'),
                        default=osp.join(default_ckpt_dir, 'medclip_resnet50.bin'),
                        # default=osp.join(default_ckpt_dir, 'resnet50_v1c-2cccc1ad.pth'),
                        help="source checkpoint file path")
    parser.add_argument('-dst',
                        # default=osp.join(default_ckpt_dir, 'mmseg_medclip_swin_tiny.pth'),
                        default=osp.join(default_ckpt_dir, 'mmseg_medclip_resnet50.pth'),
                        help="converted checkpoint file path")
    parser.add_argument('-model', default='res50', choices=('swin', 'res50'))
    return parser.parse_args()


def concat_qkv(q, k, v):
    return torch.cat([q, k, v], dim=0)


def convert_swin(ckpt: OrderedDict):
    new_ckpt = OrderedDict()
    new_ckpt['patch_embed.projection.weight'] = ckpt['vision_model.model.embeddings.patch_embeddings.projection.weight']
    new_ckpt['patch_embed.projection.bias'] = ckpt['vision_model.model.embeddings.patch_embeddings.projection.bias']
    new_ckpt['patch_embed.norm.weight'] = ckpt['vision_model.model.embeddings.norm.weight']
    new_ckpt['patch_embed.norm.bias'] = ckpt['vision_model.model.embeddings.norm.bias']

    # stage0-blocks0
    new_ckpt['stages.0.blocks.0.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.layernorm_before.weight']
    new_ckpt['stages.0.blocks.0.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.layernorm_before.bias']
    new_ckpt['stages.0.blocks.0.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.0.blocks.0.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.0.blocks.0.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.0.blocks.0.attention.self.value.weight'],
    )
    new_ckpt['stages.0.blocks.0.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.0.blocks.0.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.0.blocks.0.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.0.blocks.0.attention.self.value.bias'],
    )
    new_ckpt['stages.0.blocks.0.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.attention.output.dense.weight']
    new_ckpt['stages.0.blocks.0.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.attention.output.dense.bias']
    new_ckpt['stages.0.blocks.0.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.layernorm_after.weight']
    new_ckpt['stages.0.blocks.0.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.layernorm_after.bias']
    new_ckpt['stages.0.blocks.0.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.intermediate.dense.weight']
    new_ckpt['stages.0.blocks.0.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.intermediate.dense.bias']
    new_ckpt['stages.0.blocks.0.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.output.dense.weight']
    new_ckpt['stages.0.blocks.0.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.output.dense.bias']

    new_ckpt['stages.0.blocks.0.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.attention.self.relative_position_bias_table']
    new_ckpt['stages.0.blocks.0.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.0.attention.self.relative_position_index']

    # stage0-blocks1
    new_ckpt['stages.0.blocks.1.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.layernorm_before.weight']
    new_ckpt['stages.0.blocks.1.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.layernorm_before.bias']
    new_ckpt['stages.0.blocks.1.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.0.blocks.1.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.0.blocks.1.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.0.blocks.1.attention.self.value.weight'],
    )
    new_ckpt['stages.0.blocks.1.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.0.blocks.1.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.0.blocks.1.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.0.blocks.1.attention.self.value.bias'],
    )
    new_ckpt['stages.0.blocks.1.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.attention.output.dense.weight']
    new_ckpt['stages.0.blocks.1.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.attention.output.dense.bias']
    new_ckpt['stages.0.blocks.1.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.layernorm_after.weight']
    new_ckpt['stages.0.blocks.1.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.layernorm_after.bias']
    new_ckpt['stages.0.blocks.1.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.intermediate.dense.weight']
    new_ckpt['stages.0.blocks.1.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.intermediate.dense.bias']
    new_ckpt['stages.0.blocks.1.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.output.dense.weight']
    new_ckpt['stages.0.blocks.1.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.output.dense.bias']

    new_ckpt['stages.0.blocks.1.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.attention.self.relative_position_bias_table']
    new_ckpt['stages.0.blocks.1.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.0.blocks.1.attention.self.relative_position_index']

    # down-sample
    new_ckpt['stages.0.downsample.reduction.weight'] = ckpt[
        'vision_model.model.encoder.layers.0.downsample.reduction.weight']
    new_ckpt['stages.0.downsample.norm.weight'] = ckpt['vision_model.model.encoder.layers.0.downsample.norm.weight']
    new_ckpt['stages.0.downsample.norm.bias'] = ckpt['vision_model.model.encoder.layers.0.downsample.norm.bias']

    # stage1-blocks0
    new_ckpt['stages.1.blocks.0.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.layernorm_before.weight']
    new_ckpt['stages.1.blocks.0.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.layernorm_before.bias']
    new_ckpt['stages.1.blocks.0.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.1.blocks.0.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.1.blocks.0.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.1.blocks.0.attention.self.value.weight'],
    )
    new_ckpt['stages.1.blocks.0.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.1.blocks.0.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.1.blocks.0.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.1.blocks.0.attention.self.value.bias'],
    )
    new_ckpt['stages.1.blocks.0.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.attention.output.dense.weight']
    new_ckpt['stages.1.blocks.0.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.attention.output.dense.bias']
    new_ckpt['stages.1.blocks.0.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.layernorm_after.weight']
    new_ckpt['stages.1.blocks.0.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.layernorm_after.bias']
    new_ckpt['stages.1.blocks.0.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.intermediate.dense.weight']
    new_ckpt['stages.1.blocks.0.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.intermediate.dense.bias']
    new_ckpt['stages.1.blocks.0.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.output.dense.weight']
    new_ckpt['stages.1.blocks.0.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.output.dense.bias']

    new_ckpt['stages.1.blocks.0.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.attention.self.relative_position_bias_table']
    new_ckpt['stages.1.blocks.0.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.0.attention.self.relative_position_index']

    # stage1-blocks1
    new_ckpt['stages.1.blocks.1.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.layernorm_before.weight']
    new_ckpt['stages.1.blocks.1.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.layernorm_before.bias']
    new_ckpt['stages.1.blocks.1.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.1.blocks.1.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.1.blocks.1.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.1.blocks.1.attention.self.value.weight'],
    )
    new_ckpt['stages.1.blocks.1.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.1.blocks.1.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.1.blocks.1.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.1.blocks.1.attention.self.value.bias'],
    )
    new_ckpt['stages.1.blocks.1.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.attention.output.dense.weight']
    new_ckpt['stages.1.blocks.1.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.attention.output.dense.bias']
    new_ckpt['stages.1.blocks.1.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.layernorm_after.weight']
    new_ckpt['stages.1.blocks.1.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.layernorm_after.bias']
    new_ckpt['stages.1.blocks.1.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.intermediate.dense.weight']
    new_ckpt['stages.1.blocks.1.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.intermediate.dense.bias']
    new_ckpt['stages.1.blocks.1.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.output.dense.weight']
    new_ckpt['stages.1.blocks.1.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.output.dense.bias']

    new_ckpt['stages.1.blocks.1.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.attention.self.relative_position_bias_table']
    new_ckpt['stages.1.blocks.1.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.1.blocks.1.attention.self.relative_position_index']

    # down-sample
    new_ckpt['stages.1.downsample.reduction.weight'] = ckpt[
        'vision_model.model.encoder.layers.1.downsample.reduction.weight']
    new_ckpt['stages.1.downsample.norm.weight'] = ckpt['vision_model.model.encoder.layers.1.downsample.norm.weight']
    new_ckpt['stages.1.downsample.norm.bias'] = ckpt['vision_model.model.encoder.layers.1.downsample.norm.bias']

    # stage2-blocks0
    new_ckpt['stages.2.blocks.0.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.layernorm_before.weight']
    new_ckpt['stages.2.blocks.0.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.layernorm_before.bias']
    new_ckpt['stages.2.blocks.0.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.0.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.0.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.0.attention.self.value.weight'],
    )
    new_ckpt['stages.2.blocks.0.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.0.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.0.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.0.attention.self.value.bias'],
    )
    new_ckpt['stages.2.blocks.0.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.attention.output.dense.weight']
    new_ckpt['stages.2.blocks.0.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.attention.output.dense.bias']
    new_ckpt['stages.2.blocks.0.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.layernorm_after.weight']
    new_ckpt['stages.2.blocks.0.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.layernorm_after.bias']
    new_ckpt['stages.2.blocks.0.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.intermediate.dense.weight']
    new_ckpt['stages.2.blocks.0.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.intermediate.dense.bias']
    new_ckpt['stages.2.blocks.0.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.output.dense.weight']
    new_ckpt['stages.2.blocks.0.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.output.dense.bias']

    new_ckpt['stages.2.blocks.0.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.attention.self.relative_position_bias_table']
    new_ckpt['stages.2.blocks.0.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.0.attention.self.relative_position_index']

    # stage2-blocks1
    new_ckpt['stages.2.blocks.1.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.layernorm_before.weight']
    new_ckpt['stages.2.blocks.1.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.layernorm_before.bias']
    new_ckpt['stages.2.blocks.1.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.1.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.1.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.1.attention.self.value.weight'],
    )
    new_ckpt['stages.2.blocks.1.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.1.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.1.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.1.attention.self.value.bias'],
    )
    new_ckpt['stages.2.blocks.1.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.attention.output.dense.weight']
    new_ckpt['stages.2.blocks.1.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.attention.output.dense.bias']
    new_ckpt['stages.2.blocks.1.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.layernorm_after.weight']
    new_ckpt['stages.2.blocks.1.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.layernorm_after.bias']
    new_ckpt['stages.2.blocks.1.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.intermediate.dense.weight']
    new_ckpt['stages.2.blocks.1.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.intermediate.dense.bias']
    new_ckpt['stages.2.blocks.1.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.output.dense.weight']
    new_ckpt['stages.2.blocks.1.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.output.dense.bias']

    new_ckpt['stages.2.blocks.1.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.attention.self.relative_position_bias_table']
    new_ckpt['stages.2.blocks.1.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.1.attention.self.relative_position_index']

    # stage2-blocks2
    new_ckpt['stages.2.blocks.2.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.layernorm_before.weight']
    new_ckpt['stages.2.blocks.2.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.layernorm_before.bias']
    new_ckpt['stages.2.blocks.2.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.2.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.2.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.2.attention.self.value.weight'],
    )
    new_ckpt['stages.2.blocks.2.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.2.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.2.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.2.attention.self.value.bias'],
    )
    new_ckpt['stages.2.blocks.2.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.attention.output.dense.weight']
    new_ckpt['stages.2.blocks.2.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.attention.output.dense.bias']
    new_ckpt['stages.2.blocks.2.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.layernorm_after.weight']
    new_ckpt['stages.2.blocks.2.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.layernorm_after.bias']
    new_ckpt['stages.2.blocks.2.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.intermediate.dense.weight']
    new_ckpt['stages.2.blocks.2.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.intermediate.dense.bias']
    new_ckpt['stages.2.blocks.2.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.output.dense.weight']
    new_ckpt['stages.2.blocks.2.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.output.dense.bias']

    new_ckpt['stages.2.blocks.2.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.attention.self.relative_position_bias_table']
    new_ckpt['stages.2.blocks.2.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.2.attention.self.relative_position_index']

    # stage2-blocks3
    new_ckpt['stages.2.blocks.3.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.layernorm_before.weight']
    new_ckpt['stages.2.blocks.3.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.layernorm_before.bias']
    new_ckpt['stages.2.blocks.3.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.3.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.3.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.3.attention.self.value.weight'],
    )
    new_ckpt['stages.2.blocks.3.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.3.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.3.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.3.attention.self.value.bias'],
    )
    new_ckpt['stages.2.blocks.3.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.attention.output.dense.weight']
    new_ckpt['stages.2.blocks.3.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.attention.output.dense.bias']
    new_ckpt['stages.2.blocks.3.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.layernorm_after.weight']
    new_ckpt['stages.2.blocks.3.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.layernorm_after.bias']
    new_ckpt['stages.2.blocks.3.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.intermediate.dense.weight']
    new_ckpt['stages.2.blocks.3.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.intermediate.dense.bias']
    new_ckpt['stages.2.blocks.3.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.output.dense.weight']
    new_ckpt['stages.2.blocks.3.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.output.dense.bias']

    new_ckpt['stages.2.blocks.3.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.attention.self.relative_position_bias_table']
    new_ckpt['stages.2.blocks.3.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.3.attention.self.relative_position_index']

    # stage2-blocks4
    new_ckpt['stages.2.blocks.4.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.layernorm_before.weight']
    new_ckpt['stages.2.blocks.4.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.layernorm_before.bias']
    new_ckpt['stages.2.blocks.4.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.4.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.4.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.4.attention.self.value.weight'],
    )
    new_ckpt['stages.2.blocks.4.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.4.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.4.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.4.attention.self.value.bias'],
    )
    new_ckpt['stages.2.blocks.4.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.attention.output.dense.weight']
    new_ckpt['stages.2.blocks.4.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.attention.output.dense.bias']
    new_ckpt['stages.2.blocks.4.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.layernorm_after.weight']
    new_ckpt['stages.2.blocks.4.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.layernorm_after.bias']
    new_ckpt['stages.2.blocks.4.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.intermediate.dense.weight']
    new_ckpt['stages.2.blocks.4.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.intermediate.dense.bias']
    new_ckpt['stages.2.blocks.4.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.output.dense.weight']
    new_ckpt['stages.2.blocks.4.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.output.dense.bias']

    new_ckpt['stages.2.blocks.4.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.attention.self.relative_position_bias_table']
    new_ckpt['stages.2.blocks.4.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.4.attention.self.relative_position_index']

    # stage2-blocks5
    new_ckpt['stages.2.blocks.5.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.layernorm_before.weight']
    new_ckpt['stages.2.blocks.5.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.layernorm_before.bias']
    new_ckpt['stages.2.blocks.5.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.5.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.5.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.2.blocks.5.attention.self.value.weight'],
    )
    new_ckpt['stages.2.blocks.5.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.2.blocks.5.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.5.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.2.blocks.5.attention.self.value.bias'],
    )
    new_ckpt['stages.2.blocks.5.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.attention.output.dense.weight']
    new_ckpt['stages.2.blocks.5.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.attention.output.dense.bias']
    new_ckpt['stages.2.blocks.5.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.layernorm_after.weight']
    new_ckpt['stages.2.blocks.5.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.layernorm_after.bias']
    new_ckpt['stages.2.blocks.5.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.intermediate.dense.weight']
    new_ckpt['stages.2.blocks.5.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.intermediate.dense.bias']
    new_ckpt['stages.2.blocks.5.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.output.dense.weight']
    new_ckpt['stages.2.blocks.5.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.output.dense.bias']

    new_ckpt['stages.2.blocks.5.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.attention.self.relative_position_bias_table']
    new_ckpt['stages.2.blocks.5.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.2.blocks.5.attention.self.relative_position_index']

    # down-sample
    new_ckpt['stages.2.downsample.reduction.weight'] = ckpt[
        'vision_model.model.encoder.layers.2.downsample.reduction.weight']
    new_ckpt['stages.2.downsample.norm.weight'] = ckpt['vision_model.model.encoder.layers.2.downsample.norm.weight']
    new_ckpt['stages.2.downsample.norm.bias'] = ckpt['vision_model.model.encoder.layers.2.downsample.norm.bias']

    # stage3-blocks0
    new_ckpt['stages.3.blocks.0.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.layernorm_before.weight']
    new_ckpt['stages.3.blocks.0.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.layernorm_before.bias']
    new_ckpt['stages.3.blocks.0.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.3.blocks.0.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.3.blocks.0.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.3.blocks.0.attention.self.value.weight'],
    )
    new_ckpt['stages.3.blocks.0.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.3.blocks.0.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.3.blocks.0.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.3.blocks.0.attention.self.value.bias'],
    )
    new_ckpt['stages.3.blocks.0.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.attention.output.dense.weight']
    new_ckpt['stages.3.blocks.0.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.attention.output.dense.bias']
    new_ckpt['stages.3.blocks.0.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.layernorm_after.weight']
    new_ckpt['stages.3.blocks.0.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.layernorm_after.bias']
    new_ckpt['stages.3.blocks.0.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.intermediate.dense.weight']
    new_ckpt['stages.3.blocks.0.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.intermediate.dense.bias']
    new_ckpt['stages.3.blocks.0.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.output.dense.weight']
    new_ckpt['stages.3.blocks.0.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.output.dense.bias']

    new_ckpt['stages.3.blocks.0.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.attention.self.relative_position_bias_table']
    new_ckpt['stages.3.blocks.0.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.0.attention.self.relative_position_index']

    # stage3-blocks1
    new_ckpt['stages.3.blocks.1.norm1.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.layernorm_before.weight']
    new_ckpt['stages.3.blocks.1.norm1.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.layernorm_before.bias']
    new_ckpt['stages.3.blocks.1.attn.w_msa.qkv.weight'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.3.blocks.1.attention.self.query.weight'],
        ckpt['vision_model.model.encoder.layers.3.blocks.1.attention.self.key.weight'],
        ckpt['vision_model.model.encoder.layers.3.blocks.1.attention.self.value.weight'],
    )
    new_ckpt['stages.3.blocks.1.attn.w_msa.qkv.bias'] = concat_qkv(
        ckpt['vision_model.model.encoder.layers.3.blocks.1.attention.self.query.bias'],
        ckpt['vision_model.model.encoder.layers.3.blocks.1.attention.self.key.bias'],
        ckpt['vision_model.model.encoder.layers.3.blocks.1.attention.self.value.bias'],
    )
    new_ckpt['stages.3.blocks.1.attn.w_msa.proj.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.attention.output.dense.weight']
    new_ckpt['stages.3.blocks.1.attn.w_msa.proj.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.attention.output.dense.bias']
    new_ckpt['stages.3.blocks.1.norm2.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.layernorm_after.weight']
    new_ckpt['stages.3.blocks.1.norm2.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.layernorm_after.bias']
    new_ckpt['stages.3.blocks.1.ffn.layers.0.0.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.intermediate.dense.weight']
    new_ckpt['stages.3.blocks.1.ffn.layers.0.0.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.intermediate.dense.bias']
    new_ckpt['stages.3.blocks.1.ffn.layers.1.weight'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.output.dense.weight']
    new_ckpt['stages.3.blocks.1.ffn.layers.1.bias'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.output.dense.bias']

    new_ckpt['stages.3.blocks.1.attn.w_msa.relative_position_bias_table'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.attention.self.relative_position_bias_table']
    new_ckpt['stages.3.blocks.1.attn.w_msa.relative_position_index'] = ckpt[
        'vision_model.model.encoder.layers.3.blocks.1.attention.self.relative_position_index']

    # final norm
    new_ckpt['norm.weight'] = ckpt['vision_model.model.layernorm.weight']
    new_ckpt['norm.bias'] = ckpt['vision_model.model.layernorm.bias']
    return new_ckpt


def convert_resnet(ckpt: OrderedDict):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if ('num_batches_tracked' in k) or ('text_model' in k) or (k == 'logit_scale'):
            continue
        if 'text_model' in k:
            continue
        new_k = k.replace('vision_model.model.', '')
        new_ckpt[new_k] = v
    return new_ckpt


def main():
    args = parse_args()
    ckpt = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if args.model == 'swin':
        new_ckpt = convert_swin(ckpt)
    else:
        new_ckpt = convert_resnet(ckpt)

    mkdir_or_exist(osp.dirname(args.dst))
    torch.save(new_ckpt, args.dst)


if __name__ == '__main__':
    main()
