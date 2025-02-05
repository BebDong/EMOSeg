# coding=utf-8

import os.path as osp
import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CamVidDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
                 'sign_symbol', 'fence', 'car', 'pedestrian', 'bicyclist'),
        palette=[[128, 128, 128], [128, 0, 0], [192, 192, 128],
                 [128, 64, 128], [0, 0, 192], [128, 128, 0],
                 [192, 128, 128], [64, 64, 128], [64, 0, 128],
                 [64, 64, 0], [0, 128, 192]]
    )

    def __init__(self, ann_file, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super(CamVidDataset, self).__init__(ann_file=ann_file, img_suffix=img_suffix,
                                            seg_map_suffix=seg_map_suffix, **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], self.backend_args) and osp.isfile(self.ann_file)
