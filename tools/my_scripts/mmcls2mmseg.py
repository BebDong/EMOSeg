import argparse
import os.path as osp
from collections import OrderedDict
import torch
from mmengine.runner import CheckpointLoader

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str, help="source checkpoint file path")
parser.add_argument('dst', type=str, help="path of converted checkpoint file to save")
args = parser.parse_args()


def main():
    checkpoint = CheckpointLoader.load_checkpoint(osp.join(args.src), map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_ckpt = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('head'):
            continue
        k = k.replace('backbone.', '')
        if 'qkv.weight' in k:
            new_k = k.replace('qkv.weight', 'attn.in_proj_weight')
        elif 'qkv.bias' in k:
            new_k = k.replace('qkv.bias', 'attn.in_proj_bias')
        elif 'proj.weight' in k:
            new_k = k.replace('proj.weight', 'attn.out_proj.weight')
        elif 'proj.bias' in k:
            new_k = k.replace('proj.bias', 'attn.out_proj.bias')
        elif 'ln1.weight' == k or 'ln1.bias' == k:
            continue
        else:
            new_k = k
        new_ckpt[new_k] = v
    torch.save(new_ckpt, osp.join(args.dst))


if __name__ == '__main__':
    main()
