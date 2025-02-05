import os.path as osp
from mmengine import fileio
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NYUv2Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
                 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling',
                 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain',
                 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp',
                 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'),
        palette=[[255, 20, 23], [255, 102, 17], [255, 136, 68], [255, 238, 85], [254, 254, 56],
                 [255, 255, 153], [170, 204, 34], [187, 221, 119], [200, 207, 130], [146, 167, 126],
                 [85, 153, 238], [0, 136, 204], [34, 102, 136], [23, 82, 121], [85, 119, 119],
                 [221, 187, 51], [211, 167, 109], [169, 131, 75], [118, 118, 118], [81, 87, 74],
                 [68, 124, 105], [116, 196, 147], [142, 140, 109], [228, 191, 128], [233, 215, 142],
                 [226, 151, 93], [241, 150, 112], [225, 101, 82], [201, 74, 83], [190, 81, 104],
                 [163, 73, 116], [153, 55, 103], [101, 56, 125], [78, 36, 114], [145, 99, 182],
                 [226, 121, 163], [224, 89, 139], [124, 159, 176], [86, 152, 196], [154, 191, 136]])

    def __init__(self, ann_file, img_suffix='.jpg', seg_map_suffix='.png',
                 reduce_zero_label=True, **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label, ann_file=ann_file, **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], self.backend_args) and osp.isfile(self.ann_file)
