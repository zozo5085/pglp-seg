import argparse
import json
import os
import sys

import clip
import torch
from PIL import Image
from tqdm import tqdm

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file
from utils.preprocess import read_file_list, prepare_dataset_cls_tokens


OUTPUT_NAMES = {
    "voc": "voc_pseudo_label.json",
    "ade": "ade_pseudo_label.json",
    "cityscapes": "cityscapes_pseudo_label.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image-level pseudo-label JSON for PGLP-Seg training."
    )
    parser.add_argument("--cfg", required=True, help="Training config file.")
    parser.add_argument(
        "--out",
        default="",
        help="Output JSONL path. If empty, use text/{dataset}_pseudo_label.json.",
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument(
        "--top-k",
        default=0,
        type=int,
        help="Number of classes per image. If 0, use cfg.DATASET.K.",
    )
    parser.add_argument(
        "--window-divisor",
        default=6.0,
        type=float,
        help="Sliding-window side length is cfg.DATASET.CROP_SIZE[0] / window_divisor.",
    )
    return parser.parse_args()


def make_positions(length, window, step):
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, step))
    last = length - window
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def output_path_for(cfg, out_arg):
    if out_arg:
        return out_arg
    dataset = str(cfg.DATASET.NAME).lower()
    if dataset not in OUTPUT_NAMES:
        raise ValueError(f"Unsupported dataset for default pseudo-label name: {dataset}")
    return os.path.join("text", OUTPUT_NAMES[dataset])


def generate(cfg, out_path, batch_size, top_k, window_divisor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _, _, train_images, train_labels, _, _, _, _ = read_file_list(cfg, load_pseudo=False)
    _, _ = prepare_dataset_cls_tokens(cfg)

    text_features = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location="cpu").to(device).float()
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    k = int(top_k) if int(top_k) > 0 else int(cfg.DATASET.K)
    window = max(1, int(round(float(cfg.DATASET.CROP_SIZE[0]) / float(window_divisor))))
    step = max(1, window // 2)

    with open(out_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for image_path in tqdm(train_images, total=len(train_images)):
                img = Image.open(image_path).convert("RGB")
                width, height = img.size
                xs = make_positions(width, window, step)
                ys = make_positions(height, window, step)

                crops = []
                counts = {}
                for y in ys:
                    for x in xs:
                        x2 = min(x + window, width)
                        y2 = min(y + window, height)
                        x1 = max(0, x2 - window)
                        y1 = max(0, y2 - window)
                        crops.append(clip_preprocess(img.crop((x1, y1, x2, y2))))

                for start in range(0, len(crops), int(batch_size)):
                    batch = torch.stack(crops[start:start + int(batch_size)], dim=0).to(device)
                    image_features = clip_model.encode_image(batch)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    logits = clip_model.logit_scale.exp() * image_features.float() @ text_features.t()
                    pred = torch.argmax(logits, dim=1).detach().cpu().tolist()
                    for cls_idx in pred:
                        cls_idx = int(cls_idx)
                        counts[cls_idx] = counts.get(cls_idx, 0) + 1

                if counts:
                    labels = [
                        int(cls_idx)
                        for cls_idx, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:k]
                    ]
                else:
                    labels = []

                f.write(json.dumps(labels))
                f.write("\n")

    if sum(1 for _ in open(out_path, "r", encoding="utf-8")) != len(train_images):
        raise RuntimeError("Pseudo-label line count does not match training image count.")
    print(f"saved: {out_path}")


def main():
    args = parse_args()
    cfg = cfg_from_file(args.cfg)
    out_path = output_path_for(cfg, args.out)
    generate(cfg, out_path, args.batch_size, args.top_k, args.window_divisor)


if __name__ == "__main__":
    main()
