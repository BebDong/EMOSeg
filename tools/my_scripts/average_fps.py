import time
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmengine.registry import init_default_scope

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="config file")
parser.add_argument('--shape', type=int, nargs='+', default=[64, 64], help="image size")
parser.add_argument('--warm', type=int, default=500, help="warm-up iterations")
parser.add_argument('--iters', type=int, default=1000, help="interations")
args = parser.parse_args()


class ModelWrapper(nn.Module):
    def __init__(self, mmseg_cfg, input_shape):
        super().__init__()
        mmseg_model = MODELS.build(mmseg_cfg.model)
        self.net = revert_sync_batchnorm(mmseg_model)

        result = dict()
        result['ori_shape'] = input_shape
        result['pad_shape'] = input_shape
        self.batch_data_samples = [SegDataSample(metainfo=result)]

    def forward(self, x):
        x = self.net.extract_feat(x)
        x = self.net.decode_head(x, self.batch_data_samples)
        return x


def main():
    torch.backends.cudnn.benchmark = True

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3,) + tuple(args.shape)
    else:
        raise ValueError("Valid input value.")

    cfg = Config.fromfile(osp.join(args.config))
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.model.data_preprocessor.size = input_shape[1:]
    model = ModelWrapper(mmseg_cfg=cfg, input_shape=input_shape[2:])
    model.eval()

    img = torch.rand(size=input_shape)
    if torch.cuda.is_available():
        img = img.cuda()
        model = model.cuda()

    print(f"Warm-up for {args.warm} iterations...")
    torch.cuda.synchronize()
    for _ in range(args.warm):
        _ = model(img)
        torch.cuda.synchronize()

    print(f"Speed test for {args.iters} iterations...")
    time_spent = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(img)
        torch.cuda.synchronize()
        time_spent.append(time.perf_counter() - t_start)
    torch.cuda.synchronize()
    elapsed_time = float(np.sum(time_spent))
    print('Elapsed time: [%.2f s / %d iter]' % (elapsed_time, args.iters))
    print('Speed Time: %.2f ms / iter    FPS: %.2f' % (
        elapsed_time / args.iters * 1000, args.iters / elapsed_time))


if __name__ == '__main__':
    main()
