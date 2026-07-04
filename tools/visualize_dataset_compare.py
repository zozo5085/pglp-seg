import os
import re
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys
cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file
from utils.preprocess import read_file_list


# -----------------------------
# Palettes
# -----------------------------
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CITY_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

CITY_COLORS = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
    [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
    [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
    [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
    [  0, 80,100], [  0,  0,230], [119, 11, 32],
], dtype=np.uint8)


def voc_palette_raw():
    """VOC palette: index 0 background, 1..20 classes."""
    palette = []
    for j in range(256):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        palette.append([r, g, b])
    return np.array(palette, dtype=np.uint8)


VOC_RAW_PALETTE = voc_palette_raw()


def generic_palette(num_classes, seed=123):
    rng = np.random.default_rng(seed)
    colors = rng.integers(30, 235, size=(num_classes, 3), dtype=np.uint8)

    # make first several classes stable and visually distinct
    base = np.array([
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
        [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
        [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
    ], dtype=np.uint8)
    n = min(len(base), num_classes)
    colors[:n] = base[:n]
    return colors


# -----------------------------
# File indexing
# -----------------------------
def canonical_id_from_name(name):
    """
    Robustly extract dataset image id from filenames.

    Handles:
      pglp_seg_2011_003055.pt -> 2011_003055
      2008_000149.pt -> 2008_000149
      ADE_val_00000001.pt -> ADE_val_00000001
      000000000139.pt -> 000000000139
      frankfurt_000000_000294_leftImg8bit.pt -> frankfurt_000000_000294
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]

    m = re.search(r"[a-zA-Z]+_\d{6}_\d{6}", stem)
    if m:
        return m.group(0)

    m = re.search(r"\d{4}_\d{6}", stem)
    if m:
        return m.group(0)

    m = re.search(r"ADE_(?:train|val)_\d{8}", stem)
    if m:
        return m.group(0)

    m = re.search(r"\d{12}", stem)
    if m:
        return m.group(0)

    # remove common suffixes
    for suf in [
        "_leftImg8bit", "_gtFine_labelTrainIds", "_gtFine_labelIds",
        "_27labelTrainIds", "_labelTrainIds", "_labelIds",
    ]:
        if stem.endswith(suf):
            stem = stem[:-len(suf)]

    return stem


def build_pt_index(pred_dir, recursive=True):
    index = {}
    if pred_dir is None or pred_dir == "":
        return index

    pred_dirs = [p.strip() for p in str(pred_dir).split(",") if p.strip()]
    for one_dir in pred_dirs:
        one_dir = os.path.normpath(one_dir)
        walker = os.walk(one_dir)
        for dirpath, _, files in walker:
            for fn in files:
                if not fn.lower().endswith(".pt"):
                    continue

                full = os.path.join(dirpath, fn)
                stem = os.path.splitext(fn)[0]
                cid = canonical_id_from_name(stem)

                index[cid] = full
                index[stem] = full
            if not recursive:
                break

    return index


# -----------------------------
# Label / prediction processing
# -----------------------------
def read_file_list_safe(cfg):
    try:
        return read_file_list(cfg, load_pseudo=False)
    except TypeError:
        return read_file_list(cfg)


def load_prediction(pt_path, out_hw=None, num_classes=None):
    x = torch.load(pt_path, map_location="cpu")

    if isinstance(x, dict):
        for k in ["pred", "prediction", "mask", "output", "logits", "probs", "prob"]:
            if k in x:
                x = x[k]
                break

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported prediction type: {type(x)} in {pt_path}")

    x = x.detach().cpu().float().squeeze()

    # [C,H,W] logits/probs
    if x.ndim == 3:
        if num_classes is not None and x.shape[0] != int(num_classes) and x.shape[-1] == int(num_classes):
            x = x.permute(2, 0, 1)

        if out_hw is not None and tuple(x.shape[-2:]) != tuple(out_hw):
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=tuple(out_hw),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        pred = torch.argmax(x, dim=0).long().numpy()
        return pred.astype(np.int64)

    # [H,W] label map
    if x.ndim == 2:
        pred = x.long().numpy()
        if out_hw is not None and tuple(pred.shape) != tuple(out_hw):
            img = Image.fromarray(pred.astype(np.int32), mode="I")
            img = img.resize((int(out_hw[1]), int(out_hw[0])), resample=Image.Resampling.NEAREST)
            pred = np.asarray(img).astype(np.int64)
        return pred.astype(np.int64)

    raise ValueError(f"Unsupported prediction shape: {tuple(x.shape)} in {pt_path}")


def prepare_gt_zero_based(label_raw, dataset_name, reduce_zero_label, num_classes):
    """
    Return zero-based GT labels:
      0..C-1 valid classes, 255 ignore.
    """
    lab = np.asarray(label_raw).astype(np.int64).copy()
    name = str(dataset_name).lower()

    if name == "voc":
        # VOC raw: 0 background, 1..20 classes, 255 ignore
        out = np.full_like(lab, 255, dtype=np.int64)
        valid = (lab >= 1) & (lab <= int(num_classes))
        out[valid] = lab[valid] - 1
        return out

    if bool(reduce_zero_label):
        # ADE / Context-style: raw 0 ignore, raw 1..C -> 0..C-1
        lab[lab == 0] = 255
        lab = lab - 1
        lab[lab == 254] = 255
        lab[(lab < 0) | (lab >= int(num_classes))] = 255
        return lab

    # City / COCO-Stuff-27-style: already 0..C-1, 255 ignore
    lab[(lab < 0) | (lab >= int(num_classes))] = 255
    return lab


def get_palette(dataset_name, num_classes):
    name = str(dataset_name).lower()

    if name in ["city", "cityscapes", "gtav"]:
        return CITY_COLORS.copy()

    if name == "voc":
        # For zero-based VOC class ids 0..19, use raw VOC colors 1..20.
        return VOC_RAW_PALETTE[1:int(num_classes)+1].copy()

    return generic_palette(int(num_classes), seed=123)


def colorize_zero_based(mask, palette):
    mask = np.asarray(mask).astype(np.int64)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (mask != 255) & (mask >= 0) & (mask < len(palette))
    out[valid] = palette[mask[valid]]
    out[~valid] = np.array([0, 0, 0], dtype=np.uint8)
    return out


def overlay(image_rgb, mask_zero, palette, alpha=0.55):
    base = np.asarray(image_rgb).astype(np.float32)
    color = colorize_zero_based(mask_zero, palette).astype(np.float32)

    valid = (mask_zero != 255)
    out = base.copy()
    out[valid] = (1.0 - alpha) * base[valid] + alpha * color[valid]
    return np.clip(out, 0, 255).astype(np.uint8)


def error_map(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != 255
    correct = valid & (gt == pred)
    wrong = valid & (gt != pred)

    out[correct] = [40, 180, 80]   # green
    out[wrong] = [230, 50, 50]     # red
    return out


def save_png(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def add_title(img, title, height=34):
    img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + height), "white")
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def resize_to_same_height(arr, target_h):
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    w, h = img.size
    new_w = max(1, int(round(w * target_h / h)))
    img = img.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(img)


def make_panel(items, out_path, target_h=230):
    """
    items: list of (title, np_rgb)
    """
    panels = []
    for title, arr in items:
        arr = resize_to_same_height(arr, target_h)
        arr = add_title(arr, title)
        panels.append(arr)

    gap = 10
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    total_h = max(p.shape[0] for p in panels)
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    x = 0
    for p in panels:
        canvas[:p.shape[0], x:x+p.shape[1]] = p
        x += p.shape[1] + gap

    save_png(canvas, out_path)


def parse_id_list(s):
    if s is None or str(s).strip() == "":
        return None

    s = str(s).strip()
    if os.path.exists(s):
        ids = []
        with open(s, "r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if item:
                    ids.append(canonical_id_from_name(item))
        return set(ids)

    ids = [canonical_id_from_name(x.strip()) for x in s.split(",") if x.strip()]
    return set(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="dataset test yaml")
    ap.add_argument("--baseline-dir", required=True, help="Prediction directory, or comma-separated directories.")
    ap.add_argument("--ours-dir", required=True, help="Prediction directory, or comma-separated directories.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ids", default="", help="comma list or txt file. Empty means first --num valid samples.")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--make-summary", action="store_true")
    ap.add_argument("--no-recursive-pred", action="store_true", help="Index only the first level of each prediction directory.")
    ap.add_argument(
        "--clip-voc-gt-foreground",
        action="store_true",
        help="VOC qualitative mode only: hide prediction pixels outside GT foreground. Do not use for evaluation.",
    )
    args = ap.parse_args()

    cfg = cfg_from_file(args.cfg)
    dataset_name = str(cfg.DATASET.NAME).lower()
    num_classes = int(cfg.DATASET.NUM_CLASSES)
    reduce_zero_label = bool(cfg.DATASET.REDUCE_ZERO_LABEL)

    data = read_file_list_safe(cfg)
    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = data

    baseline_index = build_pt_index(args.baseline_dir, recursive=not args.no_recursive_pred)
    ours_index = build_pt_index(args.ours_dir, recursive=not args.no_recursive_pred)

    print("dataset:", dataset_name)
    print("num_classes:", num_classes)
    print("reduce_zero_label:", reduce_zero_label)
    print("val images:", len(val_images))
    print("baseline pt:", len(baseline_index))
    print("ours pt:", len(ours_index))

    wanted_ids = parse_id_list(args.ids)
    palette = get_palette(dataset_name, num_classes)

    made = []
    missing = 0

    for idx, (img_path, lab_path) in enumerate(zip(val_images, val_labels)):
        image_id = canonical_id_from_name(img_path)

        if wanted_ids is not None and image_id not in wanted_ids:
            continue

        b_path = baseline_index.get(image_id, None)
        o_path = ours_index.get(image_id, None)

        if b_path is None or o_path is None:
            missing += 1
            continue

        image = Image.open(img_path).convert("RGB")
        label_raw = np.asarray(Image.open(lab_path)).astype(np.int64)
        gt = prepare_gt_zero_based(
            label_raw,
            dataset_name=dataset_name,
            reduce_zero_label=reduce_zero_label,
            num_classes=num_classes,
        )

        h, w = gt.shape
        b_pred = load_prediction(b_path, out_hw=(h, w), num_classes=num_classes)
        o_pred = load_prediction(o_path, out_hw=(h, w), num_classes=num_classes)

        # Clamp invalid predicted labels
        b_pred[(b_pred < 0) | (b_pred >= num_classes)] = 255
        o_pred[(o_pred < 0) | (o_pred >= num_classes)] = 255

        if args.clip_voc_gt_foreground:
            if dataset_name != "voc":
                raise ValueError("--clip-voc-gt-foreground is only valid for VOC qualitative figures.")
            fg = gt != 255
            b_pred = b_pred.copy()
            o_pred = o_pred.copy()
            b_pred[~fg] = 255
            o_pred[~fg] = 255

        out_dir = os.path.join(args.out_root, dataset_name, image_id)
        os.makedirs(out_dir, exist_ok=True)

        img_np = np.asarray(image)
        gt_color = colorize_zero_based(gt, palette)
        b_color = colorize_zero_based(b_pred, palette)
        o_color = colorize_zero_based(o_pred, palette)

        gt_ov = overlay(img_np, gt, palette, alpha=args.alpha)
        b_ov = overlay(img_np, b_pred, palette, alpha=args.alpha)
        o_ov = overlay(img_np, o_pred, palette, alpha=args.alpha)

        b_err = error_map(gt, b_pred)
        o_err = error_map(gt, o_pred)

        save_png(img_np, os.path.join(out_dir, "image.png"))
        save_png(gt_color, os.path.join(out_dir, "gt_color.png"))
        save_png(b_color, os.path.join(out_dir, "baseline_color.png"))
        save_png(o_color, os.path.join(out_dir, "ours_color.png"))
        save_png(gt_ov, os.path.join(out_dir, "gt_overlay.png"))
        save_png(b_ov, os.path.join(out_dir, "baseline_overlay.png"))
        save_png(o_ov, os.path.join(out_dir, "ours_overlay.png"))
        save_png(b_err, os.path.join(out_dir, "baseline_error.png"))
        save_png(o_err, os.path.join(out_dir, "ours_error.png"))

        make_panel(
            [
                ("Image", img_np),
                ("GT", gt_ov),
                ("Baseline", b_ov),
                ("Ours", o_ov),
                ("Base Err", b_err),
                ("Ours Err", o_err),
            ],
            os.path.join(out_dir, "panel.png"),
        )

        made.append(image_id)
        print(f"[{len(made)}] {dataset_name}/{image_id}")

        if wanted_ids is None and len(made) >= int(args.num):
            break

    print("done.")
    print("made:", len(made))
    print("missing pairs:", missing)
    print("out:", args.out_root)

    if args.make_summary and len(made) > 0:
        # create one large contact sheet with first panel.png from each sample
        panel_paths = [
            os.path.join(args.out_root, dataset_name, image_id, "panel.png")
            for image_id in made
        ]
        panels = [np.asarray(Image.open(p).convert("RGB")) for p in panel_paths if os.path.exists(p)]

        if len(panels) > 0:
            thumb_h = 180
            thumbs = [resize_to_same_height(p, thumb_h) for p in panels]
            gap = 12
            total_w = max(t.shape[1] for t in thumbs)
            total_h = sum(t.shape[0] for t in thumbs) + gap * (len(thumbs) - 1)
            sheet = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

            y = 0
            for t in thumbs:
                sheet[y:y+t.shape[0], :t.shape[1]] = t
                y += t.shape[0] + gap

            save_png(sheet, os.path.join(args.out_root, f"{dataset_name}_summary.png"))
            print("summary:", os.path.join(args.out_root, f"{dataset_name}_summary.png"))


if __name__ == "__main__":
    main()
