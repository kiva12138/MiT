"""
Data sanity-check & visualization for the referring-segmentation datasets.

It loads samples through the same REFER API used during training, verifies that
images and masks are readable and consistent, prints an integrity report, and
saves a visualization montage (image | ground-truth mask overlay + referring
expression) to the output directory (default: ./images).

Examples
--------
    # check + visualize RefCOCO (unc) val split, 6 samples
    python check_data.py --data_root ../data --dataset refcoco --splitBy unc --split val --num_samples 6

    # check every available dataset (one montage each)
    python check_data.py --data_root ../data --dataset all
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# refer.py reads `Data_path` from Config at import time; we override it below so
# this script can point at an arbitrary data root without editing Config.py.
import refer as refer_module


# Default (dataset, splitBy, split) triples to visualize for `--dataset all`.
DEFAULT_TARGETS = [
    ('refcoco',  'unc',    'val'),
    ('refcoco+', 'unc',    'val'),
    ('refcocog', 'umd',    'val'),
    ('refclef',  'unc',    'val'),
]


def overlay_mask(image_rgb, mask, color=(255, 0, 0), alpha=0.5):
    """Return an RGB uint8 array with `mask` blended over `image_rgb`."""
    img = np.asarray(image_rgb).astype(np.float32)
    m = (mask > 0)[..., None]
    color = np.array(color, dtype=np.float32)
    blended = img * (1 - alpha * m) + color * (alpha * m)
    return blended.clip(0, 255).astype(np.uint8)


def check_one(data_root, dataset, splitBy, split, num_samples, out_dir, seed=0):
    refer_module.Data_path = data_root            # override module-level path
    from refer import REFER                        # import after override

    print(f"\n=== {dataset} (splitBy={splitBy}, split={split}) ===")
    refer = REFER(dataset, splitBy)

    if not os.path.isdir(refer.IMAGE_DIR):
        print(f"  [SKIP] image dir not found: {refer.IMAGE_DIR}")
        return None

    ref_ids = refer.getRefIds(split=split)
    if len(ref_ids) == 0:
        print(f"  [SKIP] no refs for split '{split}'")
        return None

    img_ids = refer.getImgIds(ref_ids)
    n_sents = sum(len(refer.Refs[r]['sentences']) for r in ref_ids)
    print(f"  refs: {len(ref_ids)} | images: {len(set(img_ids))} | sentences: {n_sents} "
          f"| avg sents/ref: {n_sents / len(ref_ids):.2f}")

    # ---- integrity check on a random subset ----
    rng = random.Random(seed)
    check_ids = rng.sample(ref_ids, min(200, len(ref_ids)))
    n_missing_img, n_empty_mask, n_shape_mismatch, mask_ratios = 0, 0, 0, []
    for rid in check_ids:
        ref = refer.loadRefs(rid)[0]
        img_info = refer.Imgs[ref['image_id']]
        img_path = os.path.join(refer.IMAGE_DIR, img_info['file_name'])
        if not os.path.isfile(img_path):
            n_missing_img += 1
            continue
        m = np.array(refer.getMask(ref)['mask'])
        if m.sum() == 0:
            n_empty_mask += 1
        if m.shape[:2] != (img_info['height'], img_info['width']):
            n_shape_mismatch += 1
        else:
            mask_ratios.append(float(m.sum()) / (m.shape[0] * m.shape[1]))
    print(f"  checked {len(check_ids)} refs -> missing images: {n_missing_img}, "
          f"empty masks: {n_empty_mask}, mask/image shape mismatch: {n_shape_mismatch}")
    if mask_ratios:
        print(f"  mask area ratio: mean {np.mean(mask_ratios):.3f}, "
              f"min {np.min(mask_ratios):.3f}, max {np.max(mask_ratios):.3f}")

    # ---- visualization montage ----
    vis_ids = rng.sample(ref_ids, min(num_samples, len(ref_ids)))
    n = len(vis_ids)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    for row, rid in enumerate(vis_ids):
        ref = refer.loadRefs(rid)[0]
        img_info = refer.Imgs[ref['image_id']]
        img = Image.open(os.path.join(refer.IMAGE_DIR, img_info['file_name'])).convert('RGB')
        mask = np.array(refer.getMask(ref)['mask'])
        sents = [s['raw'].strip() for s in ref['sentences']]
        title = sents[0] if sents else '(no expression)'
        if len(sents) > 1:
            title += f"   (+{len(sents) - 1} more)"

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"image  (id={ref['image_id']})", fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(overlay_mask(img, mask))
        axes[row, 1].set_title(title, fontsize=9)
        axes[row, 1].axis('off')

    fig.suptitle(f"{dataset} / {splitBy} / {split}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"data_samples_{dataset}_{splitBy}_{split}.png")
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved visualization -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                        help='root containing COCOtrain2014/, refcoco/, refcoco+/, refcocog/, ...')
    parser.add_argument('--dataset', default='refcoco',
                        help="dataset name, or 'all' to run every default target")
    parser.add_argument('--splitBy', default='unc')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', default=6, type=int)
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'images'))
    parser.add_argument('--seed', default=0, type=int)
    opt = parser.parse_args()

    data_root = os.path.abspath(opt.data_root)
    out_dir = os.path.abspath(opt.out_dir)
    print(f"data_root = {data_root}")
    print(f"out_dir   = {out_dir}")

    if opt.dataset == 'all':
        targets = DEFAULT_TARGETS
    else:
        targets = [(opt.dataset, opt.splitBy, opt.split)]

    saved = []
    for dataset, splitBy, split in targets:
        try:
            p = check_one(data_root, dataset, splitBy, split, opt.num_samples, out_dir, opt.seed)
            if p:
                saved.append(p)
        except Exception as e:
            print(f"  [ERROR] {dataset}/{splitBy}/{split}: {type(e).__name__}: {e}")

    print(f"\nDone. {len(saved)} visualization(s) saved.")


if __name__ == '__main__':
    main()
