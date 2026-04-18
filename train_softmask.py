# coding=gb2312
import os
import sys
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import options
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from utils.loader import get_training_data, get_validation_data
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import torch
import utils
import torch.nn.functional as F

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
        """
        自定义Dice系数计算
        """
        # 确保张量在正确的设备上
        pred = pred.to(self.device)
        target = target.to(self.device)

        # 如果预测值不是二值，应用阈值
        if pred.max() > 1 or pred.min() < 0:
            pred = torch.sigmoid(pred)

        # 应用阈值进行二值化
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()

        # 计算交集和并集
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()

        # 避免除零
        dice = (2. * intersection) / (union + 1e-8)

        return dice

    def item(self):
        # 为了兼容性
        return 1.0

warnings.filterwarnings('ignore')

# ---------------------------- parser #----------------------------##
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)
opt = options.Options().init(argparse.ArgumentParser(description='shadow removal')).parse_args()
print(opt)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# ---------------------------- Logs dir #----------------------------##
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'
logname = os.path.join(log_dir, log_filename)
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# #---------------------------- Set Seeds #----------------------------##
random.seed(2742)
np.random.seed(2742)
torch.manual_seed(2742)
torch.cuda.manual_seed_all(2742)

# ---------------------------- Model #----------------------------##
torch.backends.cudnn.benchmark = True
model_restoration = utils.get_arch(opt)
with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

# ---------------------------- Optimizer #----------------------------##
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

# ---------------------------- DataParallel #----------------------------##
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

# ---------------------------- Resume #----------------------------##
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1

# #---------------------------- Scheduler #----------------------------##
if opt.warmup:
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

# ---------------------------- Loss and Metrics #----------------------------##
from losses import binary_cross_entropy_loss

# 使用Dice系数替代IoU，Dice支持连续值
dice_metric = Dice(average='micro', threshold=0.5).cuda()

# ---------------------------- DataLoader #----------------------------##
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()

# ---------------------------- train #----------------------------##
print('\n')
print('================== -Now Start Epoch {} End Epoch {}- ==============='.format(start_epoch, opt.nepoch))
best_dice = 0
best_epoch = 0
best_iter = 0
eval_now = 1000
loss_scaler = NativeScaler()
torch.cuda.empty_cache()
eval_time = 0

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_dice = 0
    train_id = 1

    train_pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                      desc=f'Epoch {epoch}/{opt.nepoch}', ncols=100,
                      bar_format='{l_bar}{bar:5}{r_bar}')

    for i, data in train_pbar:
        # zero_grad
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        eval_time += 1

        if opt.use_amp:
            with torch.amp.autocast('cuda'):
                restored = model_restoration(input_)
                restored = torch.sigmoid(restored)  # 转换为0-1概率

                loss = binary_cross_entropy_loss(restored, target)
        else:
            restored = model_restoration(input_)
            restored = torch.sigmoid(restored)  # 转换为0-1概率

            loss = binary_cross_entropy_loss(restored, target)

        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())

        with torch.no_grad():
            # 使用Dice系数
            batch_dice = dice_metric(restored, target).item()
            epoch_dice += batch_dice

        epoch_loss += loss.item()

        avg_loss = epoch_loss / (i + 1)
        avg_dice = epoch_dice / (i + 1)
        current_lr = scheduler.get_lr()[0]

        train_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Dice': f'{avg_dice:.4f}',
            'LR': f'{current_lr:.6f}'
        })

        # ---------------------------- Evaluation ---------------------------- #
        if (eval_time + 1) % eval_now == 0:
            print("~=====>Evaluation time~ ")

            current_desc = train_pbar.desc
            current_postfix = train_pbar.postfix

            train_pbar.refresh()

            with torch.no_grad():
                model_restoration.eval()
                dice_val = []
                val_pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                                desc='Validating', ncols=100, leave=False,
                                bar_format='{l_bar}{bar:20}{r_bar}')

                for ii, data_val in val_pbar:
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    filenames = data_val[3]

                    B, C, H, W = target.shape
                    img_multiple_of = 16 * opt.win_size
                    H_pad = ((H + img_multiple_of) // img_multiple_of) * img_multiple_of
                    W_pad = ((W + img_multiple_of) // img_multiple_of) * img_multiple_of
                    padh = H_pad - H if H % img_multiple_of != 0 else 0
                    padw = W_pad - W if W % img_multiple_of != 0 else 0
                    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                    mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

                    restored = model_restoration(input_)
                    restored = torch.sigmoid(restored)  # 转换为0-1概率
                    restored = torch.clamp(restored, 0, 1)
                    restored = restored[:, :, :H, :W]
                    current_dice = dice_metric(restored, target).item()
                    dice_val.append(current_dice)

                    val_pbar.set_postfix({
                        'Dice': f'{np.mean(dice_val):.4f}'
                    })

                val_pbar.close()
                dice_val_avg = sum(dice_val) / len(val_loader)

                if dice_val_avg > best_dice:
                    best_dice = dice_val_avg
                    best_epoch = epoch
                    best_iter = i
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_best.pth"))

                print("\n")
                print(f"[Ep {epoch} it {i}\t Val Dice: {dice_val_avg:.4f}] ")

                with open(logname, 'a') as f:
                    f.write("[Ep %d it %d\t Val Dice: %.4f\t] ----  [best_Ep %d best_it %d Best_Dice %.4f] " \
                            % (epoch, i, dice_val_avg, best_epoch, best_iter, best_dice) + '\n')

                model_restoration.train()
                torch.cuda.empty_cache()

            train_pbar.set_description(current_desc)
            train_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Dice': f'{avg_dice:.4f}',
                'LR': f'{current_lr:.6f}'
            })

    train_pbar.close()
    epoch_time = time.time() - epoch_start_time
    scheduler.step()
    final_avg_loss = epoch_loss / len(train_loader)
    final_avg_dice = epoch_dice / len(train_loader)
    current_lr = scheduler.get_lr()[0]

    print("Epoch: {}\tTime: {:.2f}s\tLoss: {:.4f}\tTrain Dice: {:.4f}\tLearningRate {:.6f}".format(
        epoch, epoch_time, final_avg_loss, final_avg_dice, current_lr))
    print("------------------------------------------------------------------")

    with open(logname, 'a') as f:
        f.write("Epoch: {}\tTime: {:.2f}s\tLoss: {:.4f}\tTrain Dice: {:.4f}\tLearningRate {:.6f}\n".format(
            epoch, epoch_time, final_avg_loss, final_avg_dice, current_lr))

    torch.save({
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))

    if epoch % opt.checkpoint == 0:
        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))

# ---------------------------- 测试部分保持不变 ----------------------------#
if epoch == opt.nepoch:
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
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save denoised images in result directory')
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
        """
        安全地将张量转换为 (H, W, C) 格式
        """
        # 转换为numpy并移除batch维度
        if tensor.dim() == 4:  # (1, C, H, W)
            arr = tensor.squeeze(0).numpy()  # (C, H, W)
        elif tensor.dim() == 3:  # (C, H, W) 或 (1, H, W)
            arr = tensor.numpy()
            if arr.shape[0] == 1:  # (1, H, W)
                arr = arr.squeeze(0)  # (H, W)
        else:
            arr = tensor.numpy()

        # 根据维度进行转置
        if arr.ndim == 3 and arr.shape[0] in [1, 3]:  # (C, H, W) 其中 C=1 或 3
            arr = arr.transpose(1, 2, 0)  # (H, W, C)
            if arr.shape[2] == 1:  # 如果是单通道，去掉通道维度或保持
                arr = arr.squeeze(2)  # 或者保留为 (H, W, 1)
        elif arr.ndim == 3 and arr.shape[2] in [1, 3]:  # 已经是 (H, W, C)
            pass  # 不需要转置
        # 其他情况保持原样

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
            """
            自定义Dice系数计算
            """
            # 确保张量在正确的设备上
            pred = pred.to(self.device)
            target = target.to(self.device)

            # 如果预测值不是二值，应用阈值
            if pred.max() > 1 or pred.min() < 0:
                pred = torch.sigmoid(pred)

            # 应用阈值进行二值化
            pred_binary = (pred > self.threshold).float()
            target_binary = (target > self.threshold).float()

            # 计算交集和并集
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum()

            # 避免除零
            dice = (2. * intersection) / (union + 1e-8)

            return dice

        def item(self):
            # 为了兼容性
            return 1.0


    dice_metric = Dice(average='micro', threshold=0.5).cuda()

    with torch.no_grad():

        for ii, data_test in enumerate(tqdm(test_loader), 0):
            # 使用安全的转置函数
            rgb_gt = safe_transpose_to_hwc(data_test[0])
            rgb_noisy = data_test[1].cuda()

            filenames = data_test[3]

            # 打印调试信息（可选）
            if ii == 0:  # 只在第一次迭代时打印
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

            # 如果rgb_restored是张量，确保它在CPU上
            if torch.is_tensor(rgb_restored):
                rgb_restored = rgb_restored.cpu()

            # 移除batch维度（如果存在）
            if rgb_restored.dim() == 4:
                rgb_restored = rgb_restored.squeeze(0)

            # 移除batch维度
            if rgb_restored.ndim == 4:
                rgb_restored = rgb_restored.squeeze(0)  # 移除batch维度

            if rgb_restored.ndim == 3:
                # 3维数组 (C, H, W) -> (H, W, C)
                # 首先检查是否是PyTorch张量
                if torch.is_tensor(rgb_restored):
                    # PyTorch张量，使用permute方法
                    rgb_restored = rgb_restored.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                else:
                    # NumPy数组，使用transpose方法
                    rgb_restored = rgb_restored.transpose((1, 2, 0))

                # Unpad the output
                rgb_restored = rgb_restored[:height, :width, :]
            elif rgb_restored.ndim == 2:
                # 2维数组 (H, W) - 直接裁剪
                rgb_restored = rgb_restored[:height, :width]
            else:
                pass

            if args.save_images:
                # 确保将张量移到CPU并转换为numpy
                if torch.is_tensor(rgb_restored):
                    rgb_restored = rgb_restored.detach().cpu().numpy()

                # 确保图像是3通道（如果是单通道，复制为3通道）
                if rgb_restored.ndim == 2:
                    # 单通道转3通道 - 使用正确的参数
                    rgb_restored = np.stack([rgb_restored] * 3, axis=-1)
                elif rgb_restored.shape[-1] == 1:
                    # 单通道扩展为3通道
                    rgb_restored = np.repeat(rgb_restored, 3, axis=-1)

                utils.save_img(rgb_restored * 255.0, os.path.join(opt.result_dir, filenames[0]))

    import os
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F


    def calculate_dice_coefficient(pred, target, threshold=0.5, smooth=1e-8):
        """
        计算Dice系数

        Args:
            pred: 预测图像 (numpy array或tensor), 范围[0,1]或[0,255]
            target: 真实图像 (numpy array或tensor), 范围[0,1]或[0,255]
            threshold: 二值化阈值
            smooth: 平滑项，避免除零

        Returns:
            dice: Dice系数
        """
        # 转换为numpy数组
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()

        # 确保是二维数组（单通道）
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

        # 归一化到[0,1]范围
        if pred.max() > 1:
            pred = pred.astype(np.float32) / 255.0
        if target.max() > 1:
            target = target.astype(np.float32) / 255.0

        # 二值化
        pred_binary = (pred > threshold).astype(np.float32)
        target_binary = (target > threshold).astype(np.float32)

        # 计算交集和并集
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)

        # 计算Dice系数
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice


    def load_and_preprocess_image(image_path, target_size=None, convert_to_single_channel=True):
        """
        加载并预处理图像

        Args:
            image_path: 图像路径
            target_size: 目标尺寸 (width, height)
            convert_to_single_channel: 是否转换为单通道

        Returns:
            image: 预处理后的图像数组
        """
        try:
            # 加载图像
            image = Image.open(image_path)

            # 转换为灰度图（单通道）
            if convert_to_single_channel and image.mode != 'L':
                image = image.convert('L')

            # 调整尺寸
            if target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            # 转换为numpy数组
            image_array = np.array(image).astype(np.float32)

            return image_array

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None


    def calculate_dice_for_folders(folder1, folder2, file_extension='.png', threshold=0.5,
                                   target_size=None, verbose=True):
        """
        计算两个文件夹中对应图像的Dice系数

        Args:
            folder1: 第一个文件夹路径（预测结果）
            folder2: 第二个文件夹路径（真实标签）
            file_extension: 文件扩展名
            threshold: 二值化阈值
            target_size: 目标尺寸 (width, height)
            verbose: 是否显示详细信息

        Returns:
            results: 包含每个文件Dice系数和统计信息的字典
        """

        # 获取两个文件夹中的文件列表
        files1 = [f for f in os.listdir(folder1) if f.endswith(file_extension)]
        files2 = [f for f in os.listdir(folder2) if f.endswith(file_extension)]

        # 找出共同的文件
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
            # 加载图像
            img1_path = os.path.join(folder1, filename)
            img2_path = os.path.join(folder2, filename)

            img1 = load_and_preprocess_image(img1_path, target_size)
            img2 = load_and_preprocess_image(img2_path, target_size)

            if img1 is None or img2 is None:
                print(f"Skipping {filename} due to loading error")
                continue

            # 确保两个图像尺寸相同
            if img1.shape != img2.shape:
                # 调整img2的尺寸以匹配img1
                from PIL import Image
                img2_pil = Image.fromarray(img2.astype(np.uint8))
                img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
                img2 = np.array(img2_pil).astype(np.float32)

            # 计算Dice系数
            dice_score = calculate_dice_coefficient(img1, img2, threshold)
            dice_scores.append(dice_score)
            file_results[filename] = dice_score

            if verbose:
                print(f"{filename}: Dice = {dice_score:.4f}")

        if len(dice_scores) == 0:
            print("No valid image pairs found for Dice calculation")
            return None

        # 计算统计信息
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

        # 打印统计结果
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
        """将结果保存到文件"""
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


    # 使用示例
    if __name__ == "__main__":
        # 设置文件夹路径
        folder1 = './results'
        folder2 = opt.GT_dir

        # 确保文件夹存在
        if not os.path.exists(folder1):
            print(f"Folder {folder1} does not exist!")
            exit(1)

        if not os.path.exists(folder2):
            print(f"Folder {folder2} does not exist!")
            exit(1)

        # 计算Dice系数
        results = calculate_dice_for_folders(
            folder1=folder1,
            folder2=folder2,
            file_extension='.png',  # 根据实际情况修改
            threshold=0.5,  # 二值化阈值
            target_size=None,  # 可选的调整尺寸，如(256, 256)
            verbose=True
        )