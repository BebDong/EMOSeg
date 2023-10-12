# Adapted from https://github.com/Beckschen/TransUNet

import os
import h5py
import numpy as np
import argparse
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.models.utils import resize

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="config file path")
parser.add_argument('--ckpt', type=str, help="model checkpoint path")
parser.add_argument('--data-dir', type=str,
                    default='/cluster/work/cvl/qutang/TransUnetSynapse')
parser.add_argument('--num-classes', type=int, default=9, help="the number of classes")
args = parser.parse_args()

transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class Synapse(Dataset):
    def __init__(self, base_dir, list_dir=None, split='test_vol', trans=None):
        assert split in ('train', 'test_vol'), f"Unknown split {split}"
        list_dir = list_dir or base_dir
        self.transform = trans  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'train_npz', slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, 'test_vol_h5', f"{vol_name}.npy.h5")
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

    @classmethod
    def from_cfg(cls, cfg):
        pass


def mmseg_model_inference(model, img, img_shape=(512, 512)):
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    out = model.extract_feat(img)
    out = model.decode_head(out)
    if model.decode_head.mask_loss:
        out = out[0]
    out = resize(out, size=img_shape, mode='bilinear', align_corners=False)
    return out


def metric_per_case(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=(256, 256)):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y = slice_.shape
            if x != patch_size[0] or y != patch_size[1]:
                slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            slice_ = torch.from_numpy(slice_).unsqueeze(0).repeat(3, 1, 1)  # 3xHxW
            # slice_ = np.repeat(np.expand_dims(slice_, axis=0), 3, axis=0)
            img = transform(slice_).unsqueeze(0).cuda()
            net.eval()
            with torch.no_grad():
                out = mmseg_model_inference(net, img, img_shape=patch_size)
                out = torch.argmax(out, dim=1).squeeze(0)  # no need to compute softmax
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = mmseg_model_inference(net, img, img_shape=patch_size)
            out = torch.argmax(out, dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(metric_per_case(prediction == i, label == i))

    return metric_list


def main():
    # read data
    dataset = Synapse(base_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print(f"\nFound {len(dataloader)} samples for evaluation.\n")

    # get model
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location='cpu')
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    # evaluate
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(dataloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=cfg.img_scale)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f'
              % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(dataset)
    for i in range(1, args.num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f'
              % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))


def count_class_index():
    all_samples = []
    dataset = Synapse(base_dir=args.data_dir, split='test_vol')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print(f"\nFound {len(dataset)} samples.\n")
    for i, data_sample in enumerate(tqdm(dataset)):
        image, label, case_name = data_sample['image'], data_sample['label'], data_sample['case_name']
        all_samples.append(label)
    all_samples = np.concatenate(all_samples, axis=0)
    print(np.unique(all_samples))


def plot_img():
    dataset = Synapse(base_dir=args.data_dir, split='test_vol')
    for i, data_sample in enumerate(dataset):
        image, label, case_name = data_sample['image'], data_sample['label'], data_sample['case_name']
        print(f"Case name={case_name}")
        slice0 = image[0]
        return slice0
        # slice0 *= 255
        # print(np.unique(slice0))
        # slice0 = Image.fromarray(slice0).convert('RGB')
        # slice0.save(f"/cluster/home/qutang/transunet_{case_name}_slice000.jpg")
        # break
    pass


def debug():
    img_path = f"{os.getenv('TMPDIR')}/synapse/img_dir/val/case0008_slice000.jpg"
    slice0 = np.array(Image.open(img_path))
    # print(np.unique(slice0))
    return transform(slice0)


if __name__ == '__main__':
    main()
    # count_class_index()
    # plot_img()
    # debug()
    # transunet_slice = plot_img()
    # transunet_slice = torch.from_numpy(np.repeat(np.expand_dims(transunet_slice, axis=0), 3, axis=0))
    # transunet_slice = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(transunet_slice)
    # print(np.unique(transunet_slice))
    # print('\n')
    #
    # mmseg_slice = debug()
    # print(np.unique(mmseg_slice.numpy()))
    # pass
