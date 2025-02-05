from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CrackDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=False,
                         ignore_index=255, **kwargs)
