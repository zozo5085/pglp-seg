import os
import sys

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import argparse
import csv
import importlib
import io
import time
from datetime import datetime

import torch
import clip
import torch.nn.functional as F
import numpy as np
from PIL import Image

from config.configs import cfg_from_file
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC21_CLASSES = ["background"] + VOC_CLASSES


def dataset_class_names(cfg, fallback_names=None):
    names = list(fallback_names or [])
    c_num = int(cfg.DATASET.NUM_CLASSES)
    if len(names) < c_num:
        names.extend([str(i) for i in range(len(names), c_num)])
    return names[:c_num]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("ViT-B/16")
clip_model = clip_model.to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default="config/voc_test_ori_cfg.yaml",
        type=str,
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        help="model name",
        default="PGLP_Seg",
        type=str,
    )
    parser.add_argument(
        "--model_module",
        default="model.pglp_seg",
        type=str,
        help=(
            "Python module that contains PGLP_Seg. "
            "Example: model.pglp_seg"
        ),
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save visualization/debug maps for selected images.",
    )
    parser.add_argument(
        "--debug-list",
        default="",
        type=str,
        help=(
            "Optional text file with one filename per line, e.g. 2007_003106. "
            "Only these images will export debug maps."
        ),
    )
    parser.add_argument(
        "--debug-max",
        default=0,
        type=int,
        help=(
            "If --save-debug is enabled and --debug-list is empty, export only "
            "the first N images. Use 0 to export all images."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-image timing for read/preprocess/model/save.",
    )
    parser.add_argument(
        "--paper-bg-thd",
        default=0.10,
        type=float,
        help=(
            "Foreground confidence threshold used only to make standard 21-class "
            "paper visualizations and standard21 diagnostic mIoU. Pixels with "
            "max foreground probability below this value are shown as background. "
            "Used when --paper-bg-mode fixed."
        ),
    )
    parser.add_argument(
        "--paper-bg-mode",
        default="fixed",
        choices=["fixed", "adaptive"],
        type=str,
        help=(
            "Background rejection mode for visualization/diagnostic only. "
            "fixed uses --paper-bg-thd. adaptive uses Otsu on each image's "
            "max foreground probability and clips it to [--paper-bg-min, --paper-bg-max]."
        ),
    )
    parser.add_argument(
        "--paper-bg-min",
        default=0.05,
        type=float,
        help="Minimum adaptive background threshold.",
    )
    parser.add_argument(
        "--paper-bg-max",
        default=0.15,
        type=float,
        help="Maximum adaptive background threshold.",
    )
    parser.add_argument(
        "--paper-bg-max-fg-ratio",
        default=0.70,
        type=float,
        help=(
            "If adaptive foreground ratio is larger than this value, increase "
            "threshold slightly to suppress foreground leakage."
        ),
    )
    parser.add_argument(
        "--paper-bg-min-fg-ratio",
        default=0.02,
        type=float,
        help=(
            "If adaptive foreground ratio is smaller than this value, decrease "
            "threshold slightly to avoid killing foreground."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        type=str,
        help=(
            "Optional root folder for this evaluation run. If empty, the script "
            "creates a unique subfolder under cfg.SAVE_DIR/eval_runs/."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default="",
        type=str,
        help=(
            "Optional readable run tag, e.g. thd005_final. A timestamp is still "
            "appended unless --no-timestamp is used."
        ),
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help=(
            "Do not append timestamp to the run folder. Use only when you are "
            "sure files in that folder are not open/locked."
        ),
    )
    parser.add_argument(
        "--no-auto-run-dir",
        action="store_true",
        help=(
            "Disable automatic unique run directory and write directly to cfg.SAVE_DIR. "
            "Not recommended for threshold sweeps because CSV files may be overwritten "
            "or locked by Excel."
        ),
    )
    args = parser.parse_args()
    return args


def ensure_dir(path):
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def sanitize_tag(tag):
    """Make a Windows-safe folder/file tag."""
    tag = str(tag).strip()
    if tag == "":
        return ""
    bad = '<>:"/\\|?*'
    for ch in bad:
        tag = tag.replace(ch, "_")
    tag = tag.replace(" ", "_")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("._")


def threshold_tag(thd):
    return f"thd{int(round(float(thd) * 100)):02d}" if float(thd) >= 1.0 else f"thd{int(round(float(thd) * 100)):03d}"


def build_run_save_dir(base_save_dir, args):
    if bool(getattr(args, "no_auto_run_dir", False)):
        return base_save_dir

    model_short = str(args.model_module).split(".")[-1]
    thd = threshold_tag(args.paper_bg_thd)

    if args.run_tag:
        tag = sanitize_tag(args.run_tag)
    else:
        tag = sanitize_tag(f"{model_short}_{thd}")

    if not bool(getattr(args, "no_timestamp", False)):
        tag = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.output_root:
        root = args.output_root
    else:
        root = os.path.join(base_save_dir, "eval_runs")

    return os.path.join(root, tag)


def load_state_dict_flexible(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    print("FIRST CKPT KEYS =", list(ckpt.keys())[:5])

    new_state = {}
    for key, value in ckpt.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        new_state[new_key] = value

    return new_state


def load_debug_names(debug_list_path):
    if debug_list_path is None or debug_list_path == "":
        return set()

    names = set()
    with open(debug_list_path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            item = os.path.splitext(os.path.basename(item))[0]
            names.add(item)

    return names


def voc_palette():
    """Pascal VOC style palette for indexed PNG masks."""
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
        palette.extend([r, g, b])
    return palette


VOC_PALETTE = voc_palette()


def save_index_png(arr, path, size_hw=None):
    """
    Save an index mask as a palette PNG.
    arr:
        [H,W] with class ids 0..19, 255 for ignore.
    size_hw:
        Optional output size as (H,W). Uses nearest resize.
    """
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=255).astype(np.uint8)

    img = Image.fromarray(arr, mode="P")
    img.putpalette(VOC_PALETTE)

    if size_hw is not None and tuple(arr.shape[:2]) != tuple(size_hw):
        out_h, out_w = int(size_hw[0]), int(size_hw[1])
        img = img.resize((out_w, out_h), resample=Image.Resampling.NEAREST)

    img.save(path)


def palette_rgb_array():
    pal = np.asarray(VOC_PALETTE, dtype=np.uint8).reshape(256, 3)
    return pal


def save_overlay_png(value_buf, mask, path, alpha=0.55, hide_background=True):
    base = Image.open(io.BytesIO(value_buf)).convert("RGB")
    base_np = np.asarray(base).astype(np.float32)

    mask = np.asarray(mask).astype(np.int64)
    if mask.shape[:2] != base_np.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
        mask_img = mask_img.resize((base_np.shape[1], base_np.shape[0]), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask_img).astype(np.int64)

    pal = palette_rgb_array()
    valid = (mask != 255)
    if hide_background:
        valid = valid & (mask != 0)

    color = pal[np.clip(mask, 0, 255)]
    out = base_np.copy()
    out[valid] = (1.0 - alpha) * out[valid] + alpha * color[valid].astype(np.float32)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def fast_hist_standard21(pred_standard, label_raw, num_classes=21, ignore_index=255):
    pred = np.asarray(pred_standard).astype(np.int64)
    lab = np.asarray(label_raw).astype(np.int64)
    valid = (lab != ignore_index) & (lab >= 0) & (lab < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(
        num_classes * lab[valid] + pred[valid],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def iou_from_hist(hist):
    hist = hist.astype(np.float64)
    denom = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iou = np.divide(np.diag(hist), denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
    return iou


def otsu_threshold_np(values, num_bins=256):
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.10

    v = np.clip(v, 0.0, 1.0)
    hist, bin_edges = np.histogram(v, bins=num_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)

    total = hist.sum()
    if total <= 0:
        return 0.10

    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg

    mean_bg_num = np.cumsum(hist * centers)
    mean_total = mean_bg_num[-1]
    mean_fg_num = mean_total - mean_bg_num

    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float(np.median(v))

    mean_bg = np.zeros_like(centers, dtype=np.float64)
    mean_fg = np.zeros_like(centers, dtype=np.float64)
    mean_bg[valid] = mean_bg_num[valid] / weight_bg[valid]
    mean_fg[valid] = mean_fg_num[valid] / weight_fg[valid]

    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    idx = int(np.argmax(between))
    return float(centers[idx])


def compute_bg_threshold(max_prob_np, args):
    mode = str(getattr(args, "paper_bg_mode", "fixed")).lower()
    if mode == "fixed":
        return float(args.paper_bg_thd)

    lo = float(getattr(args, "paper_bg_min", 0.05))
    hi = float(getattr(args, "paper_bg_max", 0.15))
    if hi < lo:
        lo, hi = hi, lo

    t = otsu_threshold_np(max_prob_np)
    t = float(np.clip(t, lo, hi))

    fg_ratio = float((max_prob_np >= t).mean())
    max_fg_ratio = float(getattr(args, "paper_bg_max_fg_ratio", 0.70))
    min_fg_ratio = float(getattr(args, "paper_bg_min_fg_ratio", 0.02))

    if fg_ratio > max_fg_ratio:
        t = min(t + 0.03, hi)
    elif fg_ratio < min_fg_ratio:
        t = max(t - 0.03, lo)

    return float(t)


def save_gray_png(arr, path, size_hw=None):
    arr = np.asarray(arr).astype(np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        arr = arr[0]

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(arr.min()) if arr.size > 0 else 0.0
    mx = float(arr.max()) if arr.size > 0 else 0.0
    if mx > mn:
        norm = (arr - mn) / (mx - mn)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)

    img = Image.fromarray((norm * 255.0).clip(0, 255).astype(np.uint8), mode="L")

    if size_hw is not None and tuple(arr.shape[:2]) != tuple(size_hw):
        out_h, out_w = int(size_hw[0]), int(size_hw[1])
        img = img.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)

    img.save(path)


def tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_map(x):
    arr = tensor_to_numpy(x)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def prepare_eval_label(label_raw, reduce_zero_label):
    label = label_raw.astype(np.int64).copy()
    label[label == 0] = 255

    if reduce_zero_label:
        label = label - 1
        label[label == 254] = 255

    return label


def compute_per_image_miou(pred, label_eval, num_classes=20, ignore_index=255):
    pred = np.asarray(pred).astype(np.int64)
    label_eval = np.asarray(label_eval).astype(np.int64)

    valid = label_eval != ignore_index
    ious = []
    per_cls = {}

    for cid in range(num_classes):
        pred_c = (pred == cid) & valid
        label_c = (label_eval == cid) & valid
        inter = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()

        if union > 0:
            iou = float(inter) / float(union)
            ious.append(iou)
            per_cls[cid] = iou

    if len(ious) == 0:
        return float("nan"), per_cls

    return float(np.mean(ious)), per_cls


def save_debug_maps(
    debug_root,
    filename,
    value_buf,
    model,
    pred_np,
    label_eval,
    ori_shape,
    pred_standard_np=None,
    label_raw=None,
    max_prob_np=None,
    bg_thd_used=None,
):
    out_dir = os.path.join(debug_root, filename)
    ensure_dir(out_dir)

    try:
        input_img = Image.open(io.BytesIO(value_buf)).convert("RGB")
        input_img.save(os.path.join(out_dir, "image.jpg"))
    except Exception as exc:
        print(f"[Debug] failed to save input image for {filename}: {exc}", flush=True)

    save_index_png(label_eval, os.path.join(out_dir, "gt_foreground_ignorebg.png"), size_hw=ori_shape)
    save_index_png(pred_np, os.path.join(out_dir, "pred_final_foreground20.png"), size_hw=ori_shape)

    if label_raw is not None:
        save_index_png(label_raw, os.path.join(out_dir, "gt_standard21.png"), size_hw=ori_shape)
        save_overlay_png(value_buf, label_raw, os.path.join(out_dir, "gt_standard21_overlay.png"))

    if pred_standard_np is not None:
        save_index_png(pred_standard_np, os.path.join(out_dir, "pred_final_standard21.png"), size_hw=ori_shape)
        save_overlay_png(value_buf, pred_standard_np, os.path.join(out_dir, "pred_final_standard21_overlay.png"))

    if max_prob_np is not None:
        save_gray_png(max_prob_np, os.path.join(out_dir, "max_foreground_prob.png"), size_hw=ori_shape)

    if bg_thd_used is not None:
        with open(os.path.join(out_dir, "bg_threshold_used.txt"), "w", encoding="utf-8") as f:
            f.write(f"{float(bg_thd_used):.6f}\n")

    save_index_png(label_eval, os.path.join(out_dir, "gt.png"), size_hw=ori_shape)
    save_index_png(pred_np, os.path.join(out_dir, "pred_final.png"), size_hw=ori_shape)

    maps = getattr(model, "sfp_debug_maps", {})
    if maps is None:
        maps = {}

    npz_dict = {}
    for key, value in maps.items():
        try:
            npz_dict[key] = tensor_to_numpy(value)
        except Exception:
            pass

    npz_dict["pred_final_fullres"] = np.asarray(pred_np)
    npz_dict["label_eval_fullres"] = np.asarray(label_eval)
    raw_model = model.module if hasattr(model, "module") else model

    maps = getattr(raw_model, "sfp_debug_maps", {})
    if maps is None:
        maps = {}

    npz_dict = {}

    for key, value in maps.items():
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            npz_dict[key] = value
        except Exception as exc:
            print(f"[Debug] failed to pack map {key}: {exc}", flush=True)

    if hasattr(raw_model, "_debug_tensors"):
        for k, v in raw_model._debug_tensors.items():
            try:
                if hasattr(v, "detach"):
                    v = v.detach()
                    if v.dim() == 4 and v.shape[0] == 1:
                        v = v[0]
                    npz_dict[k] = v.float().cpu().numpy()
            except Exception as e:
                print("[debug tensor save failed]", k, e)

    try:
        np.savez_compressed(os.path.join(out_dir, "debug_maps.npz"), **npz_dict)
        print("[Debug] saved npz:", os.path.join(out_dir, "debug_maps.npz"), list(npz_dict.keys()), flush=True)
    except Exception as exc:
        print(f"[Debug] failed to save npz for {filename}: {exc}", flush=True)

    heatmap_keys = [
        "sfp_score",
        "sfp_outlier_mask",
        "sfp_confidence",
        "sfp_margin",
        "cpsfp_delta",
        "dtlr_update_mask",
        "dtlr_reject_mask",
        "dtlr_delta",
        "attr_update_mask",
        "attr_delta",
        "attr_logit_chair",
        "attr_logit_table",
        "attr_delta_chair",
        "attr_delta_table",
    ]



    for key in heatmap_keys:
        if key in maps:
            try:
                save_gray_png(
                    squeeze_map(maps[key]),
                    os.path.join(out_dir, f"{key}.png"),
                    size_hw=ori_shape,
                )
            except Exception as exc:
                print(f"[Debug] failed heatmap {key} for {filename}: {exc}", flush=True)

    pred_keys = [
        "pred_before_cpsfp",
        "pred_after_cpsfp",
        "pred_before_dtlr",
        "pred_after_dtlr",
        "pred_before_attr",
        "pred_after_attr",
        "pred_final_lowres",
    ]

    for key in pred_keys:
        if key in maps:
            try:
                save_index_png(
                    squeeze_map(maps[key]),
                    os.path.join(out_dir, f"{key}.png"),
                    size_hw=ori_shape,
                )
            except Exception as exc:
                print(f"[Debug] failed pred map {key} for {filename}: {exc}", flush=True)


def test():
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)

    base_save_dir = cfg.SAVE_DIR
    run_save_dir = build_run_save_dir(base_save_dir, args)
    cfg.SAVE_DIR = run_save_dir

    print("CFG_FILE =", cfg_file)
    print("LOAD_PATH =", cfg.LOAD_PATH)
    print("BASE_SAVE_DIR =", base_save_dir)
    print("RUN_SAVE_DIR =", cfg.SAVE_DIR)
    print("TEST.PD =", cfg.TEST.PD)
    print("MODEL_MODULE =", args.model_module)
    print("PAPER_BG_MODE =", str(getattr(args, "paper_bg_mode", "fixed")))
    print("PAPER_BG_THD =", float(args.paper_bg_thd))
    print("PAPER_BG_MIN/MAX =", float(getattr(args, "paper_bg_min", 0.05)), float(getattr(args, "paper_bg_max", 0.15)))

    ensure_dir(cfg.SAVE_DIR)

    debug_root = os.path.join(cfg.SAVE_DIR, "debug_maps")
    if args.save_debug:
        ensure_dir(debug_root)

    debug_names = load_debug_names(args.debug_list)
    if args.save_debug:
        if len(debug_names) > 0:
            print(f"[Debug] export selected names: {len(debug_names)}")
        elif args.debug_max > 0:
            print(f"[Debug] export first {args.debug_max} images")
        else:
            print("[Debug] export ALL images. This may take disk space.")

    (
        train_filenames,
        val_filenames,
        train_images,
        train_labels,
        val_images,
        val_labels,
        results_iou,
        pseudo_classes,
    ) = read_file_list(cfg)

    results_iou = [
        os.path.join(cfg.SAVE_DIR, filename + ".pt")
        for filename in val_filenames
    ]

    cls_name_token, text = prepare_dataset_cls_tokens(cfg)
    class_names = dataset_class_names(cfg, text)
    is_voc_dataset = str(cfg.DATASET.NAME).lower() == "voc"
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location="cpu")

    model_module = importlib.import_module(args.model_module)
    ModelClass = getattr(model_module, "PGLP_Seg")

    if args.model_name == "PGLP_Seg":
        model = ModelClass(
            cfg=cfg,
            clip_model=clip_model,
            rank=0,
            zeroshot_weights=text_weight,
        )
    else:
        raise NotImplementedError(
            "This debug eval script currently supports PGLP_Seg only. "
            "Use author test.py if you need ReCLIP."
        )

    print("MODEL MODULE =", model.__class__.__module__)

    new_weight = load_state_dict_flexible(cfg.LOAD_PATH)

    model.load_state_dict(new_weight, strict=True)
    print("[Load] checkpoint loaded with strict=True")

    model = model.to(device)

    c_num = cfg.DATASET.NUM_CLASSES
    model.eval()

    sfp_ratio_rows = []
    per_image_rows = []
    success_num = 0
    debug_saved_count = 0

    hist_standard21 = np.zeros((21, 21), dtype=np.float64)

    with torch.no_grad():
        for idx in range(len(val_images)):
            t_all = time.time()

            t0 = time.time()
            with open(val_images[idx], "rb") as f:
                value_buf = f.read()
            t_read = time.time() - t0

            t0 = time.time()
            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)
            t_pre = time.time() - t0

            t0 = time.time()
            label_img = Image.open(val_labels[idx])
            label_img.load()
            ori_shape = tuple((label_img.size[1], label_img.size[0]))  # (H,W)
            label_raw = np.asarray(label_img).copy()
            label_eval = prepare_eval_label(
                label_raw,
                reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL,
            )
            t_label = time.time() - t0

            gt_cls = []
            shape = img.shape[2:]

            filename = val_filenames[idx]

            filename = val_filenames[idx]
            filename_key = os.path.splitext(os.path.basename(str(filename)))[0]

            export_debug_this = False
            if args.save_debug:
                if len(debug_names) > 0:
                    export_debug_this = filename_key in debug_names
                elif args.debug_max > 0:
                    export_debug_this = debug_saved_count < args.debug_max
                else:
                    export_debug_this = True

            setattr(model, "sfp_debug_export", bool(export_debug_this))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.time()
            output = model(
                img,
                gt_cls,
                text_weight,
                cls_name_token,
                training=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_model = time.time() - t0

            output = F.interpolate(
                output,
                shape,
                None,
                "bilinear",
                False,
            ).reshape(1, c_num, shape[0], shape[1])

            output = F.interpolate(
                output,
                ori_shape,
                None,
                "bilinear",
                False,
            ).reshape(1, c_num, ori_shape[0], ori_shape[1])

            prob = F.softmax(output, dim=1)
            max_prob, pred_fg = prob.max(dim=1)
            output = pred_fg.squeeze(dim=0)

            pred_np = output.detach().cpu().numpy().astype(np.int64)
            max_prob_np = max_prob.squeeze(dim=0).detach().cpu().numpy().astype(np.float32)

            bg_thd_used = compute_bg_threshold(max_prob_np, args)
            pred_standard_np = pred_np + 1
            pred_standard_np[max_prob_np < bg_thd_used] = 0

            hist_standard21 += fast_hist_standard21(
                pred_standard_np,
                label_raw,
                num_classes=21,
                ignore_index=255,
            )

            image_miou, image_class_iou = compute_per_image_miou(
                pred_np,
                label_eval,
                num_classes=c_num,
                ignore_index=255,
            )

            per_image_rows.append({
                "img_idx": int(idx),
                "filename": filename,
                "ori_h": int(ori_shape[0]),
                "ori_w": int(ori_shape[1]),
                "pre_h": int(shape[0]),
                "pre_w": int(shape[1]),
                "per_image_miou": image_miou,
                "num_valid_classes": int(len(image_class_iou)),
                "bg_thd_used": float(bg_thd_used),
                "bg_mode": str(getattr(args, "paper_bg_mode", "fixed")),
            })

            t0 = time.time()
            save_path = os.path.normpath(results_iou[idx])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if idx < 3:
                print("val_filename:", val_filenames[idx])
                print("save_path:", save_path)
                print("eval_path:", results_iou[idx])
            torch.save(output.cpu(), save_path)
            t_save = time.time() - t0

            if export_debug_this:
                save_debug_maps(
                    debug_root=debug_root,
                    filename=filename,
                    value_buf=value_buf,
                    model=model,
                    pred_np=pred_np,
                    label_eval=label_eval,
                    ori_shape=ori_shape,
                    pred_standard_np=pred_standard_np,
                    label_raw=label_raw,
                    max_prob_np=max_prob_np,
                    bg_thd_used=bg_thd_used,
                )
                debug_saved_count += 1

            stats_batch = getattr(model, "sfp_last_stats_batch", [])
            if stats_batch is not None and len(stats_batch) > 0:
                stat = stats_batch[0].copy()
                stat["filename"] = filename
                stat["img_idx"] = int(idx)
                stat["per_image_miou"] = image_miou
                sfp_ratio_rows.append(stat)

            success_num += 1

            if args.timing:
                print(
                    f"[TIME] idx={idx:04d} {filename} "
                    f"read={t_read:.3f}s pre={t_pre:.3f}s "
                    f"label={t_label:.3f}s model={t_model:.3f}s "
                    f"save={t_save:.3f}s total={time.time() - t_all:.3f}s "
                    f"miou={image_miou:.4f}",
                    flush=True,
                )

            if idx % 100 == 0:
                print("filenames:{}, img_idx:{}".format(filename, idx))
                print("filenames:{}, img_idx:{}".format(filename, idx))

        iou = mean_iou(
            results_iou,
            val_labels,
            num_classes=c_num + 1,
            ignore_index=255,
            nan_to_num=0,
            reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL,
        )

        print(iou["IoU"])
        avg = iou["IoU"].sum() / c_num

        per_class_csv = os.path.join(cfg.SAVE_DIR, "per_class_iou.csv")
        with open(per_class_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "iou"])
            for cid, cname in enumerate(class_names):
                writer.writerow([cid, cname, float(iou["IoU"][cid])])
        print(f"[Per-class IoU] saved to {per_class_csv}")

        standard21_avg_all = float("nan")
        standard21_avg_fg = float("nan")
        if is_voc_dataset:
            standard21_iou = iou_from_hist(hist_standard21)
            standard21_avg_all = float(np.nanmean(standard21_iou))
            standard21_avg_fg = float(np.nanmean(standard21_iou[1:]))

            standard21_csv = os.path.join(cfg.SAVE_DIR, "per_class_iou_standard21_bgthd.csv")
            with open(standard21_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["class_id", "class_name", "iou"])
                for cid, cname in enumerate(VOC21_CLASSES):
                    writer.writerow([cid, cname, float(standard21_iou[cid])])
            print(f"[Standard21 IoU diagnostic] saved to {standard21_csv}")
            print(f"[Standard21] bg_thd={float(args.paper_bg_thd):.3f}, mIoU_all21={standard21_avg_all:.4f}, mIoU_fg20={standard21_avg_fg:.4f}")
        else:
            print("[Standard21] skipped; this VOC background diagnostic is not valid for dataset:", cfg.DATASET.NAME)

        per_image_csv = os.path.join(cfg.SAVE_DIR, "per_image_iou.csv")
        with open(per_image_csv, "w", newline="") as f:
            fieldnames = [
                "img_idx",
                "filename",
                "ori_h",
                "ori_w",
                "pre_h",
                "pre_w",
                "per_image_miou",
                "num_valid_classes",
                "bg_thd_used",
                "bg_mode",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"[Per-image IoU] saved to {per_image_csv}")

        summary_csv = os.path.join(cfg.SAVE_DIR, "summary_metrics.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["foreground_only_mIoU", float(avg)])
            if is_voc_dataset:
                writer.writerow(["standard21_bg_mode", str(getattr(args, "paper_bg_mode", "fixed"))])
                writer.writerow(["standard21_bg_threshold_fixed", float(args.paper_bg_thd)])
                writer.writerow(["standard21_bg_threshold_min", float(getattr(args, "paper_bg_min", 0.05))])
                writer.writerow(["standard21_bg_threshold_max", float(getattr(args, "paper_bg_max", 0.15))])
                writer.writerow(["standard21_mIoU_all21_diagnostic", float(standard21_avg_all)])
                writer.writerow(["standard21_mIoU_fg20_diagnostic", float(standard21_avg_fg)])
            writer.writerow(["success_num", int(success_num)])
            writer.writerow(["num_images", int(len(val_images))])
            writer.writerow(["debug_saved_count", int(debug_saved_count)])
        print(f"[Summary] saved to {summary_csv}")
        print(f"[Run output dir] {cfg.SAVE_DIR}")

        print("avg:%.4f" % avg)
        print("\n\nfinish with %d/%d\nthe mIOU:%.4lf" % (success_num, len(val_images), avg))

        if len(sfp_ratio_rows) > 0:
            csv_path = os.path.join(cfg.SAVE_DIR, "urd_ratio_records.csv")

            fieldnames = sorted({
                key
                for row in sfp_ratio_rows
                for key in row.keys()
            })

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(sfp_ratio_rows)

            print(f"[URD stats] saved to {csv_path}")

            ratios = np.array([r["ratio"] for r in sfp_ratio_rows], dtype=np.float32)
            outliers = np.array([r["outliers"] for r in sfp_ratio_rows], dtype=np.float32)
            diffs = np.array([r["diff_mean"] for r in sfp_ratio_rows], dtype=np.float32)

            proxy_available = np.array(
                [r.get("proxy_available_ratio", 0.0) for r in sfp_ratio_rows],
                dtype=np.float32
            )

            print("[URD ratio summary]")
            print(f"num_images             = {len(ratios)}")
            print(f"ratio mean             = {ratios.mean():.4f}")
            print(f"ratio std              = {ratios.std():.4f}")
            print(f"ratio min              = {ratios.min():.4f}")
            print(f"ratio max              = {ratios.max():.4f}")
            print(f"outlier mean           = {outliers.mean():.2f}")
            print(f"diff mean              = {diffs.mean():.6f}")
            print(f"proxy_available mean   = {proxy_available.mean():.4f}")
            print(f"saved to               = {csv_path}")
        else:
            print("[URD ratio summary] no records collected.")


if __name__ == "__main__":
    test()
