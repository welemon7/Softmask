# coding=gb2312
import os
import argparse
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

EPS = 1e-6

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Assume img in [0,1] sRGB -> linear RGB"""
    a = 0.055
    mask = img <= 0.04045
    out = np.empty_like(img)
    out[mask] = img[mask] / 12.92
    out[~mask] = ((img[~mask] + a) / (1 + a)) ** 2.4
    return out


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    a = 0.055
    mask = img <= 0.0031308
    out = np.empty_like(img)
    out[mask] = img[mask] * 12.92
    out[~mask] = (1 + a) * (img[~mask] ** (1 / 2.4)) - a
    return out


# -------------------------
# Guided filter (He et al.) (fast)
# -------------------------
def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    I: guidance image, shape (H,W), float32
    p: filtering input (same size), float32
    r: radius
    eps: regularization
    returns q: filtered p
    """
    # using box filter (cv2.boxFilter)
    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)

    q = mean_a * I + mean_b
    return q


def resize_to_match(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = target_shape[:2]
    if img.shape[:2] != (h, w):
        if img.ndim == 3:
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img_resized
    return img


def make_soft_mask_from_pair(
        shadow_img: np.ndarray,
        gt_img: np.ndarray,
        *,
        use_linear: bool = True,
        gamma_correct: bool = True,
        eps: float = 1e-6,
        multiscale_radii: tuple = (15, 35, 75),
        percentile_clip: float = 99.0,
        sigmoid_sigma: float = 0.06,
        guided_radius: int = 8,
        guided_eps: float = 1e-3,
        chroma_consistency_thresh: float = 0.15,
        chroma_suppress: float = 0.5
) -> Dict[str, np.ndarray]:
    """
        shadow_img, gt_img : np.ndarray, shape (H,W,3), dtype float in [0,1]
        dict with keys:
          'raw_ratio'   : L_shadow / (L_gt + eps)
          'raw_mask'    : 1 - ratio (clipped)
          'softmask'  : multiscale fused mask before guided filter
    """
    assert shadow_img.shape == gt_img.shape, f"Shape mismatch: shadow {shadow_img.shape} vs gt {gt_img.shape}"
    H, W, _ = shadow_img.shape

    # linearize (physical domain)
    s = np.clip(shadow_img, 0.0, 1.0)
    g = np.clip(gt_img, 0.0, 1.0)
    if use_linear:
        s_lin = srgb_to_linear(s)
        g_lin = srgb_to_linear(g)
    else:
        s_lin = s.copy()
        g_lin = g.copy()

    # compute luminance (linear)
    Ls = 0.299 * s_lin[..., 0] + 0.587 * s_lin[..., 1] + 0.114 * s_lin[..., 2]
    Lg = 0.299 * g_lin[..., 0] + 0.587 * g_lin[..., 1] + 0.114 * g_lin[..., 2]
    # clip tiny values
    Lg = np.clip(Lg, eps, None)

    # raw ratio: how much darker the shadow is vs GT
    ratio = Ls / (Lg + eps)  # <1 in shadowed regions
    ratio = np.clip(ratio, 0.0, 1.5)  # bound (some noise)
    raw_mask = np.clip(1.0 - ratio, 0.0, 1.0)  # larger -> stronger shadow

    # OPTION: color/chroma consistency check
    # compute per-channel ratios and measure spread: if spread large, probably not shadow
    ch_ratio = (s_lin + eps) / (g_lin + eps)  # shape HWC
    ch_mean = ch_ratio.mean(axis=2)
    ch_std = ch_ratio.std(axis=2)
    # where chroma std is large (object color change), reduce mask
    chroma_factor = 1.0 - np.clip(ch_std / (chroma_consistency_thresh + eps), 0.0, 1.0)  # in [0,1]
    chroma_factor = 1.0 * (1.0 - chroma_suppress) + chroma_suppress * chroma_factor  # conservative

    raw_mask = raw_mask * chroma_factor

    # multi-scale masks: compute thresholded / smoothed masks at different Gaussian scales, then fuse
    masks = []
    for r in multiscale_radii:
        # smooth ratio by gaussian (simulate smooth illumination at this scale)
        k = max(1, int(r))
        Ls_blur = cv2.GaussianBlur(Ls.astype(np.float32), ksize=(k | 1, k | 1), sigmaX=r)
        Lg_blur = cv2.GaussianBlur(Lg.astype(np.float32), ksize=(k | 1, k | 1), sigmaX=r)
        ratio_blur = np.clip(Ls_blur / (Lg_blur + eps), 0.0, 1.5)
        m = np.clip(1.0 - ratio_blur, 0.0, 1.0)
        masks.append(m)

    # fuse masks (we use max + average to be robust)
    masks_stack = np.stack(masks, axis=0)
    fuse_max = masks_stack.max(axis=0)
    fuse_mean = masks_stack.mean(axis=0)
    softmask = np.clip(0.6 * fuse_max + 0.4 * fuse_mean, 0.0, 1.0)


    out = {
        "raw_ratio": ratio,
        "raw_mask": raw_mask,
        "softmask": softmask
    }
    return out



def load_img_f32(path, size: Optional[Tuple[int, int]] = None):
    im = Image.open(path).convert("RGB")
    original_size = im.size  # (width, height)

    if size is not None:
        im = im.resize(size, Image.LANCZOS)
        print(f"Resized {path} from {original_size} to {size}")
    else:
        print(f"Loaded {path} at original size: {original_size}")

    arr = np.array(im).astype(np.float32) / 255.0
    return arr


def save_softmask_only(soft_mask: np.ndarray, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_8bit = (np.clip(soft_mask, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(mask_8bit).save(output_path)
    print(f"Saved softmask: {output_path}")


def get_image_pairs(shadow_dir: str, gt_dir: str, shadow_exts: List[str] = None) -> List[Tuple[str, str]]:
    if shadow_exts is None:
        shadow_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    pairs = []

    shadow_files = []
    for ext in shadow_exts:
        shadow_files.extend([f for f in os.listdir(shadow_dir) if f.lower().endswith(ext)])

    shadow_files.sort()  

    for shadow_file in shadow_files:
        shadow_path = os.path.join(shadow_dir, shadow_file)

        name_without_ext = os.path.splitext(shadow_file)[0]

        gt_path = None
        for ext in shadow_exts:
            possible_gt_path = os.path.join(gt_dir, name_without_ext + ext)
            if os.path.exists(possible_gt_path):
                gt_path = possible_gt_path
                break

        if gt_path and os.path.exists(gt_path):
            pairs.append((shadow_path, gt_path, name_without_ext))
        else:
            print(f"Warning: No corresponding GT image found for {shadow_file}")

    return pairs


# -------------------------
# main
# -------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="soft shadow mask generator for batch processing")
    parser.add_argument("--shadow_dir", default=r"D:\welemon2077\my_cv_storyCVPR\_0_deep learning\dataset\WSRD+(shadow+GT)\half_size\train\shadow", help="path to directory containing shadow images")
    parser.add_argument("--gt_dir", default=r"D:\welemon2077\my_cv_storyCVPR\_0_deep learning\dataset\WSRD+(shadow+GT)\half_size\train\non_shadow", help="path to directory containing GT shadow-free images")
    parser.add_argument("--outdir", default=r"D:\welemon2077\my_cv_storyCVPR\_0_deep learning\dataset\WSRD+(shadow+GT)\half_size\train\GT_softmask", help="where to save softmasks")
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="resize to specific width and height (optional)")
    parser.add_argument("--max_size", type=int, help="resize longer side to this size (keep aspect ratio)")
    parser.add_argument("--suffix", default="", help="suffix to add to output filenames (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.shadow_dir):
        print(f"Error: Shadow directory '{args.shadow_dir}' does not exist")
        return

    if not os.path.exists(args.gt_dir):
        print(f"Error: GT directory '{args.gt_dir}' does not exist")
        return

    pairs = get_image_pairs(args.shadow_dir, args.gt_dir)

    if not pairs:
        print("No matching image pairs found!")
        return

    print(f"Found {len(pairs)} image pairs to process")

    os.makedirs(args.outdir, exist_ok=True)

    processed_count = 0
    error_count = 0

    for shadow_path, gt_path, base_name in pairs:
        try:
            print(f"\nProcessing: {base_name}")
            print(f"  Shadow: {os.path.basename(shadow_path)}")
            print(f"  GT: {os.path.basename(gt_path)}")

            if args.size:
                shadow = load_img_f32(shadow_path, size=args.size)
                gt = load_img_f32(gt_path, size=args.size)
            elif args.max_size:
                shadow_img = Image.open(shadow_path).convert("RGB")
                gt_img = Image.open(gt_path).convert("RGB")

                w, h = shadow_img.size
                if w > h:
                    new_w = args.max_size
                    new_h = int(h * args.max_size / w)
                else:
                    new_h = args.max_size
                    new_w = int(w * args.max_size / h)

                shadow = load_img_f32(shadow_path, size=(new_w, new_h))
                gt = load_img_f32(gt_path, size=(new_w, new_h))
            else:
                shadow = load_img_f32(shadow_path)
                gt = load_img_f32(gt_path)

            if shadow.shape != gt.shape:
                print(f"  Warning: Image size mismatch. Shadow: {shadow.shape}, GT: {gt.shape}")
                print("  Resizing GT to match shadow image...")
                gt = resize_to_match(gt, shadow.shape[:2])

            print(f"  Processing images with size: {shadow.shape[1]}x{shadow.shape[0]}")

            # softmask
            out = make_soft_mask_from_pair(
                shadow, gt,
                use_linear=True,
                multiscale_radii=(15, 35, 75),
                sigmoid_sigma=0.06,
                guided_radius=8,
                guided_eps=1e-3
            )

            # softmask
            output_filename = f"{base_name}{args.suffix}.png"
            output_path = os.path.join(args.outdir, output_filename)
            save_softmask_only(out['softmask'], output_path)

            processed_count += 1
            print(f"Successfully processed: {output_filename}")

        except Exception as e:
            error_count += 1
            print(f"Error processing {base_name}: {str(e)}")
            continue

    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")
    print(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main_cli()