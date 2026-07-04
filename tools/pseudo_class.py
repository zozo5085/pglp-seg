import torch
import os
import clip
import argparse
from PIL import Image
from tqdm import tqdm
import time
import os
import json
import sys
cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)
from utils.preprocess import read_file_list, prepare_dataset_cls_tokens, preprocess, val_preprocess
from config.configs import cfg_from_file
def reset_output_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    print(f"[PseudoClass] reset output file: {path}")


def count_jsonl_lines(path):
    n = 0
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

voc_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
               'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
pascal_context_classes = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                          'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
                          'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                          'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
                          'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                          'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
ade_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet',
               'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
               'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
               'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
               'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
               'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
               'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
               'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
               'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
               'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land',
               'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
               'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket',
               'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
               'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood',
               'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
               'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

coco_stuff_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building',
                      'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes',
                      'clouds', 'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor',
                      'floor', 'stone floor', 'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture',
                      'grass', 'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss',
                      'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform',
                      'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand',
                      'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural',
                      'table', 'tent', 'textile', 'towel', 'tree', 'vegetable', 'brick wall', 'concrete wall', 'wall',
                      'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window',
                      'window', 'wood']

cityscapes_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                      'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_learnable_params_with_name(model, prefix=''):
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(f'{prefix}{name}, Shape: {param.shape}')
    for name, module in model.named_children():
        print_learnable_params_with_name(module, prefix=f'{name}.' if name else '')


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_train_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='PGLP_Seg', type=str)
    parser.add_argument('--out', dest='out_file',
                        help='optional output pseudo-label json path',
                        default='', type=str)
    args = parser.parse_args()
    return args


def default_output_path(cfg):
    dataset_name = str(cfg.DATASET.NAME).lower()
    default_map = {
        "voc": "text/voc_pseudo_label.json",
        "ade": "text/ade_pseudo_label.json",
        "cityscapes": "text/cityscapes_pseudo_label.json",
        "context": "text/context_pseudo_label.json",
        "stuff": "text/coco_pseudo_label.json",
    }
    return default_map.get(dataset_name, f"text/{dataset_name}_pseudo_label.json")


def generate_pseudo_labels(images, labels, cfg, window_size, step_size, out_path):

    # Reset output file BEFORE generation.
    # Do not write temp here, because temp is created per image.
    reset_output_file(out_path)

    clip_model, new_clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)
    clip_model.eval()

    _, text = prepare_dataset_cls_tokens(cfg)

    text_features = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).to(torch.float16)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    window_size = list(map(int, window_size))
    step_size = int(step_size)

    def make_positions(length, win, step):
        """
        Cover the whole image.
        Original range(0, height - window, step) may skip small images
        or miss the last border region.
        """
        if length <= win:
            return [0]

        pos = list(range(0, length - win + 1, step))
        last = length - win
        if len(pos) == 0 or pos[-1] != last:
            pos.append(last)
        return pos

    loop = tqdm(zip(images, labels), total=len(images))
    idx = 0
    print(len(images))

    with torch.no_grad():
        for image, label_path in loop:
            idx += 1

            temp = []
            cls_dict = {}

            # Pseudo label generation does not require GT labels.
            # Skip label loading and preprocess for speed.
            img = Image.open(image).convert("RGB")
            width, height = img.size

            xs = make_positions(width, window_size[0], step_size)
            ys = make_positions(height, window_size[1], step_size)

            crop_tensors = []
            crop_boxes = []

            for y in ys:
                for x in xs:
                    x2 = min(x + window_size[0], width)
                    y2 = min(y + window_size[1], height)
                    x1 = max(0, x2 - window_size[0])
                    y1 = max(0, y2 - window_size[1])

                    box = (x1, y1, x2, y2)
                    cropped_img = img.crop(box)

                    crop_tensor = new_clip_preprocess(cropped_img)
                    crop_tensors.append(crop_tensor)
                    crop_boxes.append(box)

            if len(crop_tensors) > 0:
                crop_batch_size = 64

                for start in range(0, len(crop_tensors), crop_batch_size):
                    batch = torch.stack(
                        crop_tensors[start:start + crop_batch_size],
                        dim=0
                    ).to(device)

                    image_features = clip_model.encode_image(batch)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)

                    logit_scale = clip_model.logit_scale.exp()
                    logits_per_crop = logit_scale * image_features @ text_features.t()

                    pred_label = torch.argmax(logits_per_crop, dim=1)

                    for cls in pred_label.tolist():
                        cls = int(cls)
                        cls_dict[cls] = cls_dict.get(cls, 0) + 1

            # Use classes predicted by sliding windows.
            if len(cls_dict) > 0:
                # sort by occurrence count, high to low
                sorted_items = sorted(cls_dict.items(), key=lambda x: x[1], reverse=True)
                temp = [int(k) for k, v in sorted_items[:cfg.DATASET.K]]
            else:
                temp = []

            # Fallback: whole-image top-K
            if len(temp) == 0:
                full_img = new_clip_preprocess(img).unsqueeze(dim=0).to(device)

                image_features = clip_model.encode_image(full_img)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                logit_scale = clip_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()

                _, pred_label = torch.sort(logits_per_image, descending=True, dim=1)
                temp = pred_label[:, :cfg.DATASET.K].squeeze(dim=0).tolist()
                temp = [int(x) for x in temp]

            with open(out_path, mode='a', encoding='utf-8') as cls_json:
                cls_json.write(json.dumps(temp))
                cls_json.write('\n')

            loop.set_postfix(
                idx=idx,
                num_classes=len(temp)
            )
    print("[PseudoClass] generation finished.")

    n_out = count_jsonl_lines(out_path)
    print(f'[PseudoClass] saved {n_out}/{len(images)} lines to {out_path}')

    if n_out != len(images):
        raise RuntimeError(
            f'Pseudo output line mismatch: {n_out} != {len(images)} at {out_path}'
        )


if __name__ == '__main__':
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    if args.model_name != 'PGLP_Seg':
        raise ValueError("pseudo_class.py supports --model PGLP_Seg.")
    crop_size = cfg.DATASET.CROP_SIZE
    w = crop_size[0] / 6
    s = w / 2
    out_path = args.out_file if args.out_file else default_output_path(cfg)

    # _, _, train_images, train_labels, _, _, _, pseudo_classes = read_file_list(cfg)
    _, _, train_images, train_labels, _, _, _, _old_pseudo_classes = read_file_list(
        cfg,
        load_pseudo=False
    )

    generate_pseudo_labels(
        train_images,
        train_labels,
        cfg,
        window_size=(w, w),
        step_size=s,
        out_path=out_path,
    )
