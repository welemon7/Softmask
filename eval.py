import os
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from tqdm import tqdm
import argparse
import torch

def load_and_resize_image(image_path, target_size=(256, 256), is_mask=False):
    try:
        if is_mask:
            img = Image.open(image_path).convert('L')
        else:
            img = Image.open(image_path).convert('RGB')

        img_resized = img.resize(target_size, Image.BILINEAR)
        img_array = np.array(img_resized)
        if is_mask:
            img_array = (img_array > 128).astype(np.float32)
        else:
            img_array = img_array.astype(np.float32) / 255.0

        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def calculate_metrics_with_mask(img1, img2, mask):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if mask.shape[:2] != img1.shape[:2]:
        mask = cv2.resize(mask, (img1.shape[1], img1.shape[0]))

    mask_binary = (mask > 0.5).astype(np.float32)
    mask_ns = 1 - mask_binary

    bm = np.where(mask_binary == 0, np.zeros_like(mask_binary), np.ones_like(mask_binary))  # binarize mask
    bm = np.expand_dims(bm.squeeze(), axis=2)

    # ===================== ALL =====================
    psnr_full = psnr_loss(img1, img2, data_range=1.0)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    ssim_full = ssim_loss(gray1, gray2, data_range=1.0)

    rmse_full = np.abs(cv2.cvtColor(img1, cv2.COLOR_RGB2LAB) - cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)).mean() * 3
    # lab1 = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2LAB)
    # lab2 = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2LAB)
    # rmse_full = np.sqrt(np.mean((lab1.astype(float) - lab2.astype(float)) ** 2))

    # ===================== Shadow =====================
    if mask_binary.sum() > 0:
        # mask_binary_3ch = mask_binary[..., np.newaxis]

        img1_shadow = img1 * bm
        img2_shadow = img2 * bm
        psnr_shadow = psnr_loss(img1_shadow, img2_shadow, data_range=1.0)

        gray1_shadow = gray1 * bm.squeeze()
        gray2_shadow = gray2 * bm.squeeze()
        ssim_shadow = ssim_loss(gray1_shadow, gray2_shadow, data_range=1.0)

        rmse_shadow = np.abs(cv2.cvtColor(img1 * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(img2 * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum()
        # lab_diff = lab1.astype(float) - lab2.astype(float)
        # rmse_shadow = np.sqrt(np.mean(lab_diff[mask_binary > 0] ** 2))
    else:
        psnr_shadow = ssim_shadow = rmse_shadow = 0.0

    # ===================== Nonshadow =====================
    if mask_ns.sum() > 0:
        # mask_ns_3ch = mask_ns[..., np.newaxis]
        img1_ns = img1 * (1-bm)
        img2_ns = img2 * (1-bm)
        psnr_ns = psnr_loss(img1_ns, img2_ns, data_range=1.0)

        gray1_ns = gray1 * (1-bm.squeeze())
        gray2_ns = gray2 * (1-bm.squeeze())
        ssim_ns = ssim_loss(gray1_ns, gray2_ns, data_range=1.0)

        rmse_ns = np.abs(cv2.cvtColor(img1 * (1 - bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(img2 * (1 - bm),cv2.COLOR_RGB2LAB)).sum() / (1 - bm).sum()
    else:
        psnr_ns = ssim_ns = rmse_ns = 0.0

    return {
        'full': {'psnr': psnr_full, 'ssim': ssim_full, 'rmse': rmse_full},
        'shadow': {'psnr': psnr_shadow, 'ssim': ssim_shadow, 'rmse': rmse_shadow},
        'non_shadow': {'psnr': psnr_ns, 'ssim': ssim_ns, 'rmse': rmse_ns}
    }


def calculate_metrics(img1, img2):
    metrics = calculate_metrics_with_mask(img1, img2, np.ones_like(img1[:, :, 0]))
    return metrics['full']['psnr'], metrics['full']['ssim'], metrics['full']['rmse']


def get_image_files(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    return sorted(image_files)


def evaluate_folders(folder1, folder2, mask_folder=None, target_size=(256, 256)):
    files1 = get_image_files(folder1)
    files2 = get_image_files(folder2)

    print(f"Folder 1: {len(files1)} images")
    print(f"Folder 2: {len(files2)} images")
    if mask_folder:
        print(f"Mask folder: {mask_folder}")

    common_files = set(files1) & set(files2)
    common_files = sorted(list(common_files))

    print(f"Common images: {len(common_files)}")

    if len(common_files) == 0:
        print("No common images found!")
        return

    all_metrics = {
        'full': {'psnr': [], 'ssim': [], 'rmse': []},
        'shadow': {'psnr': [], 'ssim': [], 'rmse': []},
        'non_shadow': {'psnr': [], 'ssim': [], 'rmse': []}
    }


    for filename in tqdm(common_files, desc="Evaluating images"):
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        img1 = load_and_resize_image(img1_path, target_size)
        img2 = load_and_resize_image(img2_path, target_size)

        if img1 is None or img2 is None:
            print(f"Skipping {filename} due to loading error")
            continue

        mask = None
        if mask_folder:
            mask_path = os.path.join(mask_folder, filename)
            if os.path.exists(mask_path):
                mask = load_and_resize_image(mask_path, target_size, is_mask=True)
            else:
                print(f"Warning: Mask not found for {filename}, using full image metrics only")
                mask = np.ones_like(img1[:, :, 0])

        if mask is not None:
            metrics = calculate_metrics_with_mask(img1, img2, mask)
        else:
            psnr, ssim, rmse = calculate_metrics(img1, img2)
            metrics = {
                'full': {'psnr': psnr, 'ssim': ssim, 'rmse': rmse},
                'shadow': {'psnr': 0.0, 'ssim': 0.0, 'rmse': 0.0},
                'non_shadow': {'psnr': 0.0, 'ssim': 0.0, 'rmse': 0.0}
            }

        for region in ['full', 'shadow', 'non_shadow']:
            for metric in ['psnr', 'ssim', 'rmse']:
                all_metrics[region][metric].append(metrics[region][metric])

    if all_metrics['full']['psnr']:
        results = {}
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS (Three Region Metrics)")
        print("=" * 80)
        print(f"Total images evaluated: {len(all_metrics['full']['psnr'])}")
        print(f"Target size: {target_size}")

        regions = ['full', 'shadow', 'non_shadow']
        region_names = ['Full Image', 'Shadow Region', 'Non-Shadow Region']

        for region, region_name in zip(regions, region_names):
            avg_psnr = np.mean(all_metrics[region]['psnr'])
            avg_ssim = np.mean(all_metrics[region]['ssim'])
            avg_rmse = np.mean(all_metrics[region]['rmse'])

            print(f"\n{region_name} Metrics:")
            print(f"  PSNR (RGB):  {avg_psnr:.4f}")
            print(f"  SSIM (Gray): {avg_ssim:.4f}")
            print(f"  RMSE (LAB):  {avg_rmse:.4f}")

            results[f'{region}_psnr'] = avg_psnr
            results[f'{region}_ssim'] = avg_ssim
            results[f'{region}_rmse'] = avg_rmse

        results['num_images'] = len(all_metrics['full']['psnr'])
        return results
    else:
        print("No valid images to evaluate!")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate image quality between two folders with three region metrics')
    parser.add_argument('--folder1', type=str, required=True, help='Path to first folder containing images')
    parser.add_argument('--folder2', type=str, required=True, help='Path to second folder containing images')
    parser.add_argument('--mask_folder', type=str, default=None,
                        help='Path to folder containing mask images (optional)')
    parser.add_argument('--size', type=int, default=256, help='Target size for resizing (default: 256)')

    args = parser.parse_args()

    if not os.path.exists(args.folder1):
        print(f"Error: Folder {args.folder1} does not exist!")
        return

    if not os.path.exists(args.folder2):
        print(f"Error: Folder {args.folder2} does not exist!")
        return

    if args.mask_folder and not os.path.exists(args.mask_folder):
        print(f"Warning: Mask folder {args.mask_folder} does not exist! Using full image metrics only.")

    print(f"Comparing images between:")
    print(f"Folder 1: {args.folder1}")
    print(f"Folder 2: {args.folder2}")
    if args.mask_folder:
        print(f"Mask folder: {args.mask_folder}")
    print(f"Resize to: {args.size}x{args.size}")
    print()

    results = evaluate_folders(args.folder1, args.folder2,
                               mask_folder=args.mask_folder,
                               target_size=(args.size, args.size))

    return results


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        folder1 = r"D:\welemon2077\my_cv_storyCVPR\_4softmask\result\results_ISTD+"
        folder2 = "D:\\welemon2077\\my_cv_storyCVPR\\deep learning\\dataset\\ISTD_Adjusted\\test\\non_shadow"
        mask_folder = "D:\\welemon2077\\my_cv_storyCVPR\\deep learning\\dataset\\ISTD_Adjusted\\test\\mask"

        if os.path.exists(folder1) and os.path.exists(folder2):
            results = evaluate_folders(folder1, folder2, mask_folder=mask_folder)
        else:
            print("Please provide folder paths as arguments")
            print(
                "Usage: python evaluate_images.py --folder1 path1 --folder2 path2 [--mask_folder mask_path] [--size 256]")
    else:
        main()