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
from losses import CharbonnierLoss
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import torch
import utils
from utils.loader import get_training_data, get_validation_data
from torchmetrics import PeakSignalNoiseRatio
warnings.filterwarnings('ignore')

# ---------------------------- parser #----------------------------##
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)
opt = options.Options().init(argparse.ArgumentParser(description='shadow removal')).parse_args()
print(opt)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


# ---------------------------- Logs dir #----------------------------##
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'
logname = os.path.join(log_dir, log_filename)
# print("Now time is : ", datetime.datetime.now().isoformat())
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
#     lr = utils.load_optim(optimizer, path_chk_rest)
#
#     for p in optimizer.param_groups: p['lr'] = lr
#     warmup = False
#     new_lr = lr
#     print('------------------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:",new_lr)
#     print('------------------------------------------------------------------------------')
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)


# #---------------------------- Scheduler #----------------------------##
if opt.warmup:
    # print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    # print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

# ---------------------------- Loss #----------------------------##
criterion = CharbonnierLoss().cuda()

# ---------------------------- DataLoader #----------------------------##
# print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
# print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

# ---------------------------- train #----------------------------##
print('\n')
print('================== -Now Start Epoch {} End Epoch {}- ==============='.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = 1000
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
loss_scaler = NativeScaler()
torch.cuda.empty_cache()
eval_time = 0

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_psnr = 0
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

        if epoch > 5:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)

        if opt.use_amp:
            with torch.amp.autocast('cuda'):
                restored = model_restoration(input_, mask)
                restored = torch.clamp(restored, 0, 1)
                loss = criterion(restored, target)
        else:
            restored = model_restoration(input_, mask)
            restored = torch.clamp(restored, 0, 1)
            loss = criterion(restored, target)

        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())

        with torch.no_grad():
            batch_psnr = psnr_metric(restored, target).item()
            epoch_psnr += batch_psnr

        epoch_loss += loss.item()

        avg_loss = epoch_loss / (i + 1)
        avg_psnr = epoch_psnr / (i + 1)
        current_lr = scheduler.get_lr()[0]

        train_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'PSNR': f'{avg_psnr:.2f}',
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
                psnr_val_rgb = []
                val_pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                                desc='Validating', ncols=100, leave=False,
                                bar_format='{l_bar}{bar:20}{r_bar}')

                for ii, data_val in val_pbar:
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    filenames = data_val[3]

                    restored = model_restoration(input_, mask)
                    restored = torch.clamp(restored, 0, 1)
                    current_psnr = utils.batch_PSNR(restored, target, False).item()
                    psnr_val_rgb.append(current_psnr)

                    val_pbar.set_postfix({
                        'PSNR': f'{np.mean(psnr_val_rgb):.2f}'
                    })

                val_pbar.close()
                psnr_val_rgb = sum(psnr_val_rgb) / len(val_loader)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_best.pth"))

                print("\n")
                print(f"[Ep {epoch} it {i}\t Val PSNR: {psnr_val_rgb:.2f}] ")


                with open(logname, 'a') as f:
                    f.write("[Ep %d it %d\t Val PSNR: %.2f\t] ----  [best_Ep %d best_it %d Best_PSNR %.2f] " \
                            % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')

                model_restoration.train()
                torch.cuda.empty_cache()


            train_pbar.set_description(current_desc)
            train_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'PSNR': f'{avg_psnr:.2f}',
                'LR': f'{current_lr:.6f}'
            })


    train_pbar.close()
    epoch_time = time.time() - epoch_start_time
    scheduler.step()
    final_avg_loss = epoch_loss / len(train_loader)
    final_avg_psnr = epoch_psnr / len(train_loader)
    current_lr = scheduler.get_lr()[0]

    print("Epoch: {}\tTime: {:.2f}s\tLoss: {:.4f}\tTrain PSNR: {:.2f}\tLearningRate {:.6f}".format(
        epoch, epoch_time, final_avg_loss, final_avg_psnr, current_lr))
    print("------------------------------------------------------------------")

    with open(logname, 'a') as f:
        f.write("Epoch: {}\tTime: {:.2f}s\tLoss: {:.4f}\tTrain PSNR: {:.2f}\tLearningRate {:.6f}\n".format(
            epoch, epoch_time, final_avg_loss, final_avg_psnr, current_lr))


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
# print("Now time is : ", datetime.datetime.now().isoformat())


# ---------------------------- test #----------------------------##
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
    # from ptflops import get_model_complexity_info

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

    img_multiple_of = 8 * opt.win_size


    def resize_to_target(image, target_size=256):
        if len(image.shape) == 3:
            h, w, c = image.shape
            resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = image.shape
            resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized


    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []
        rmse_val_rgb = []
        psnr_val_s = []
        ssim_val_s = []
        psnr_val_ns = []
        ssim_val_ns = []
        rmse_val_s = []
        rmse_val_ns = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            rgb_noisy = data_test[1].cuda()
            mask = data_test[2].cuda()
            filenames = data_test[3]

            # Pad the input if not_multiple_of win_size * 8
            height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
            mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

            if args.tile is None:
                rgb_restored = model_restoration(rgb_noisy, mask)
            else:
                # test the image tile by tile
                b, c, h, w = rgb_noisy.shape
                tile = min(args.tile, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"
                tile_overlap = args.tile_overlap

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                        mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                        out_patch = model_restoration(in_patch, mask_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                        W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                rgb_restored = E.div_(W)

            rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

            # Unpad the output
            rgb_restored = rgb_restored[:height, :width, :]

            if args.save_images:
                utils.save_img(rgb_restored * 255.0, os.path.join(opt.result_dir, filenames[0]))
    from eval import evaluate_folders

    model_result = opt.result_dir
    GT = opt.GT_dir
    mask = opt.mask_dir

    folder1 = model_result
    folder2 = GT
    mask_folder = mask

    results = evaluate_folders(folder1, folder2, mask_folder=mask_folder)




