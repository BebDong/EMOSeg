import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="Config file path.")
parser.add_argument('--ckpt', type=str, help="Checkpoint file path.")
parser.add_argument('--result', type=str, default='/cluster/work/cvl/qutang/tmp',
                    help="Results saving directory path.")
parser.add_argument('--data', type=str, default='/cluster/work/cvl/qutang/TransUnetSynapse',
                    help="Dataset directory path.")
args = parser.parse_args()


def main():
    # results dir
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # model
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location='cpu')
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    # data
    valid_img_names = ('case0001_slice108', 'case0002_slice095',
                       'case0003_slice134', 'case0004_slice106')
    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    for _, data in enumerate(dataloader):
        data = model.data_preprocessor(data, True)
        print(data)
        exit()

        img = data['inputs']
        img_path = data['data_samples'][0].get('img_path')
        img_name = os.path.basename(img_path).split('.')[0]  # delete postfix
        if not (img_name in valid_img_names):
            continue
        img_size = data['data_samples'][0].get('img_shape')
        with torch.no_grad():
            out = model.extract_feat(img)
            _, masks = model.decode_head(out)
        masks = F.interpolate(masks, size=img_size, mode='bilinear', align_corners=False).squeeze()

        for i, mask in enumerate(masks):
            sns.set()
            sns.heatmap(mask.cpu().numpy(), cbar=False)
            plt.axis('off')
            mask_name = f"{img_name}_class{i}.png"
            print(f"Plot and save mask: {mask_name}")
            plt.savefig(os.path.join(args.result, mask_name), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
