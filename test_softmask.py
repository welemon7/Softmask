# coding=gb2312
import numpy as np
import os, sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from utils.loader import get_validation_data
import utils
import cv2

parser = argparse.ArgumentParser(description='ShadowRemoval')
parser.add_argument('--arch', default='PhasorFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', default=True, help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
parser.add_argument('--tile', type=int, default=None,
                    help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--eval_size', type=int, default=256, help='Size for evaluation resize')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import options

opt = options.Options().init(argparse.ArgumentParser(description='shadow removal')).parse_args()
utils.mkdir(opt.result_dir)
test_dataset = get_validation_data(opt.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, opt.weights)
print("===>Testing using weights: ", opt.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 16 * opt.win_size



def safe_transpose_to_hwc(tensor):
    if tensor.dim() == 4:  # (1, C, H, W)
        arr = tensor.squeeze(0).numpy()  # (C, H, W)
    elif tensor.dim() == 3:  # (C, H, W) (1, H, W)
        arr = tensor.numpy()
        if arr.shape[0] == 1:  # (1, H, W)
            arr = arr.squeeze(0)  # (H, W)
    else:
        arr = tensor.numpy()


    if arr.ndim == 3 and arr.shape[0] in [1, 3]:  # (C, H, W)
        arr = arr.transpose(1, 2, 0)  # (H, W, C)
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)
    elif arr.ndim == 3 and arr.shape[2] in [1, 3]:
        pass

    return arr

class Dice:
    def __init__(self, average='micro', threshold=0.5):
        self.average = average
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to(self, device):
        self.device = device
        return self

    def cuda(self):
        self.device = torch.device('cuda')
        return self

    def __call__(self, pred, target):
        return self.calculate_dice(pred, target)

    def calculate_dice(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)
        if pred.max() > 1 or pred.min() < 0:
            pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        dice = (2. * intersection) / (union + 1e-8)

        return dice

    def item(self):
        return 1.0

dice_metric = Dice(average='micro', threshold=0.5).cuda()

with torch.no_grad():

    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = safe_transpose_to_hwc(data_test[0])
        rgb_noisy = data_test[1].cuda()

        filenames = data_test[3]

        if ii == 0:
            print(f"Input tensor shapes:")
            print(f"rgb_gt shape: {data_test[0].shape}")
            print(f"Processed rgb_gt shape: {rgb_gt.shape}")

        # Pad the input if not_multiple_of win_size * 8
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')


        if args.tile is None:
            rgb_restored = model_restoration(rgb_noisy)
            rgb_restored = torch.sigmoid(rgb_restored)

        else:
            # test the image tile by tile
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 16 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]

                    out_patch = model_restoration(in_patch)
                    out_patch = torch.sigmoid(out_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            rgb_restored = E.div_(W)
            rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

            # Unpad the output
            rgb_restored = rgb_restored[:height, :width, :]

        if torch.is_tensor(rgb_restored):
            rgb_restored = rgb_restored.cpu()

        if rgb_restored.dim() == 4:
            rgb_restored = rgb_restored.squeeze(0)

        if rgb_restored.ndim == 4:
            rgb_restored = rgb_restored.squeeze(0)


        if rgb_restored.ndim == 3:
            if torch.is_tensor(rgb_restored):
                rgb_restored = rgb_restored.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            else:
                rgb_restored = rgb_restored.transpose((1, 2, 0))

            # Unpad the output
            rgb_restored = rgb_restored[:height, :width, :]
        elif rgb_restored.ndim == 2:
            rgb_restored = rgb_restored[:height, :width]
        else:
            pass

        if args.save_images:
            if torch.is_tensor(rgb_restored):
                rgb_restored = rgb_restored.detach().cpu().numpy()
            if rgb_restored.ndim == 2:
                rgb_restored = np.stack([rgb_restored] * 3, axis=-1)
            elif rgb_restored.shape[-1] == 1:
                rgb_restored = np.repeat(rgb_restored, 3, axis=-1)

            utils.save_img(rgb_restored * 255.0, os.path.join(opt.result_dir, filenames[0]))

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def calculate_dice_coefficient(pred, target, threshold=0.5, smooth=1e-8):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    if pred.ndim == 3:
        if pred.shape[0] == 1 or pred.shape[0] == 3:
            pred = pred.squeeze()
        elif pred.shape[2] == 1 or pred.shape[2] == 3:
            pred = pred.squeeze()

    if target.ndim == 3:
        if target.shape[0] == 1 or target.shape[0] == 3:
            target = target.squeeze()
        elif target.shape[2] == 1 or target.shape[2] == 3:
            target = target.squeeze()

    if pred.max() > 1:
        pred = pred.astype(np.float32) / 255.0
    if target.max() > 1:
        target = target.astype(np.float32) / 255.0

    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice


def load_and_preprocess_image(image_path, target_size=None, convert_to_single_channel=True):
    try:
        image = Image.open(image_path)
        if convert_to_single_channel and image.mode != 'L':
            image = image.convert('L')

        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        image_array = np.array(image).astype(np.float32)

        return image_array

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def calculate_dice_for_folders(folder1, folder2, file_extension='.png', threshold=0.5,
                               target_size=None, verbose=True):
    files1 = [f for f in os.listdir(folder1) if f.endswith(file_extension)]
    files2 = [f for f in os.listdir(folder2) if f.endswith(file_extension)]

    common_files = set(files1) & set(files2)

    if len(common_files) == 0:
        print(f"No common files found between {folder1} and {folder2}")
        return None

    if verbose:
        print(f"Found {len(common_files)} common files")
        print(f"Files in folder1: {len(files1)}")
        print(f"Files in folder2: {len(files2)}")
        print(f"Common files: {len(common_files)}")

    dice_scores = []
    file_results = {}

    for filename in sorted(common_files):
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        img1 = load_and_preprocess_image(img1_path, target_size)
        img2 = load_and_preprocess_image(img2_path, target_size)

        if img1 is None or img2 is None:
            print(f"Skipping {filename} due to loading error")
            continue

        if img1.shape != img2.shape:
            from PIL import Image
            img2_pil = Image.fromarray(img2.astype(np.uint8))
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
            img2 = np.array(img2_pil).astype(np.float32)

        dice_score = calculate_dice_coefficient(img1, img2, threshold)
        dice_scores.append(dice_score)
        file_results[filename] = dice_score

        if verbose:
            print(f"{filename}: Dice = {dice_score:.4f}")

    if len(dice_scores) == 0:
        print("No valid image pairs found for Dice calculation")
        return None

    dice_scores = np.array(dice_scores)
    results = {
        'file_results': file_results,
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'max_dice': np.max(dice_scores),
        'min_dice': np.min(dice_scores),
        'median_dice': np.median(dice_scores),
        'num_files': len(dice_scores)
    }

    if verbose:
        print("\n" + "=" * 50)
        print("Dice Coefficient Results Summary")
        print("=" * 50)
        print(f"Number of images: {results['num_files']}")
        print(f"Mean Dice: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
        print(f"Max Dice: {results['max_dice']:.4f}")
        print(f"Min Dice: {results['min_dice']:.4f}")
        print(f"Median Dice: {results['median_dice']:.4f}")
        print("=" * 50)

    return results


def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Dice Coefficient Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of images: {results['num_files']}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}\n")
        f.write(f"Max Dice: {results['max_dice']:.4f}\n")
        f.write(f"Min Dice: {results['min_dice']:.4f}\n")
        f.write(f"Median Dice: {results['median_dice']:.4f}\n\n")

        f.write("Per-file results:\n")
        for filename, dice in results['file_results'].items():
            f.write(f"{filename}: {dice:.4f}\n")


if __name__ == "__main__":
    folder1 = './results'
    folder2 = opt.GT_dir

    if not os.path.exists(folder1):
        print(f"Folder {folder1} does not exist!")
        exit(1)

    if not os.path.exists(folder2):
        print(f"Folder {folder2} does not exist!")
        exit(1)

    results = calculate_dice_for_folders(
        folder1=folder1,
        folder2=folder2,
        file_extension='.png',
        threshold=0.5,
        target_size=None,
        verbose=True
    )