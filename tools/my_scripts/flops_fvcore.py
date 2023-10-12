import argparse
import os.path as osp
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmengine.registry import init_default_scope
from fvcore.nn import FlopCountAnalysis, parameter_count_table

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="config file")
parser.add_argument('--shape', type=int, nargs='+', default=[1024, 1024], help="image size")
args = parser.parse_args()


class ModelWrapper(nn.Module):
    def __init__(self, mmseg_cfg, input_shape=(512, 1024)):
        super().__init__()
        mmseg_model = MODELS.build(mmseg_cfg.model)
        self.net = revert_sync_batchnorm(mmseg_model)

        # result = dict()
        # result['ori_shape'] = input_shape
        # result['pad_shape'] = input_shape
        # self.batch_data_samples = [SegDataSample(metainfo=result)]

        self.batch_img_metas = [{'img_shape': input_shape}]

    def forward(self, x):
        x = self.net.encode_decode(x, self.batch_img_metas)
        return x


def main():
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

    flops = FlopCountAnalysis(model, img)
    print("FLOPs: {:.2e}".format(flops.total()))
    print(parameter_count_table(model))


if __name__ == '__main__':
    main()
