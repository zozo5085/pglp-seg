import argparse
import torch
import clip
from torch import optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import ssl
import os
import sys
import csv
import numpy as np
cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
ssl._create_default_https_context = ssl._create_unverified_context
from config.configs import cfg_from_file
from model.pglp_seg import PGLP_Seg
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_train_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='PGLP_Seg', type=str)
    args = parser.parse_args()
    return args


class Train(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_filenames, _, self.train_images, self.train_labels, _, _, _, self.pseudo_classes = read_file_list(cfg)

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()
        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(rank, world_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    log = open('experiments/log_voc_rectification.txt', mode='a')
    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(cfg)
    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)

    train_data = Train(cfg)
    train_loader = DataLoader(dataset=train_data, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True, batch_size=cfg.TRAIN.BATCH_SIZE, collate_fn=custom_collate_fn)
    

    model = PGLP_Seg(cfg=cfg, clip_model=clip_model, rank=device, zeroshot_weights=text_weight)
    model = model.to(device)
    raw_model = model.module if hasattr(model, "module") else model

    print("\n[Trainable parameters]")
    for name, p in raw_model.named_parameters():
        if p.requires_grad:
            print(name, tuple(p.shape))

    print("\n[Frozen CLIP / ViT check]")
    bad = []
    for name, p in raw_model.named_parameters():
        if (
                name.startswith("clip.")
                or name.startswith("vit.")
                or name.startswith("proj.")
                or name.startswith("logit_scale")
        ):
            if p.requires_grad:
                bad.append(name)

    if len(bad) == 0:
        print("CLIP / ViT / proj / logit_scale are frozen.")
    else:
        print("WARNING: these frozen parameters are trainable:")
        for n in bad:
            print(n)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9,
                          weight_decay=0.0005)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    if cfg.TRAIN.EPOCH >= 0:
        stop_epoch = cfg.TRAIN.EPOCH
    else:
        stop_epoch = max_epoch
    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = 0.0
    for epoch in range(max_epoch):
        idx = 0
        model.train()
        running_loss = 0.0

        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR, power=0.9)
        loop = tqdm(train_loader)

        for img, label, img_metas, filenames, pseudo_class in loop:
            time.sleep(0.08)
            gt_cls = []
            batch_size = img.shape[0]
            for i in range(batch_size):
                temp = [int(tensor) if isinstance(tensor, int) else int(tensor.item()) for tensor in pseudo_class[i]]
                gt_cls.append(temp)

                if len(temp) == 0:
                    continue
            if len(gt_cls[0]) == 0:
                continue
            output, loss = model(img.to(device), gt_cls, text_weight, cls_name_token, training=True, img_metas=img_metas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_postfix(epoch=epoch, img_loss=loss.item(), avg_loss=running_loss / (idx + 1))
            if idx % 100 == 0:
                print(
                    'filenames:{}, img_idx:{}, img_loss:{:.5f}, avg_loss:{:.5f}'.format(
                        filenames, idx, loss.item(), running_loss / (idx + 1)
                    ),
                    file=log
                )
            idx += 1
        print('epoch {} finish, lr:{}'.format(epoch, lr), file=log)

        if rank == 0:
            model.eval()
            sfp_ratio_rows = []
            success_num = 0
            with torch.no_grad():
                for idx in range(len(val_images)):
                    with open(val_images[idx], 'rb') as f:
                        value_buf = f.read()
                    img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)
                    label = Image.open(val_labels[idx])
                    ori_shape = tuple((label.size[1], label.size[0]))
                    label = np.asarray(label)
                    gt_cls = []
                    label_cls = set(label.flatten().tolist()[1:])
                    for cls in label_cls:
                        if cls != 0 and cls != 255:
                            gt_cls.append(cls - 1)
                    shape = img.shape[2:]
                    output = model(img.to(device), gt_cls, text_weight, cls_name_token, training=False)

                    N, C, H, W = output.shape

                    if args.model_name == "PGLP_Seg":
                        _output = F.softmax(output * 10, dim=1)
                        max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                        selected_cls = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(
                            N, C, H, W
                        )
                        output[selected_cls] = -100

                    output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
                    output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])

                    output = F.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1).squeeze(dim=0)
                    save_path = os.path.normpath(results_iou[idx])
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(output.cpu(), save_path)
                    inner_model = model.module if hasattr(model, "module") else model
                    stats_batch = getattr(inner_model, "sfp_last_stats_batch", [])

                    if stats_batch is not None and len(stats_batch) > 0:
                        stat = stats_batch[0].copy()
                        stat["filename"] = val_filenames[idx]
                        stat["img_idx"] = int(idx)
                        sfp_ratio_rows.append(stat)
                    success_num += 1
                    if idx % 100 == 0:
                        print('filenames:{}, img_idx:{}'.format(val_filenames[idx], idx))

                dataset_name = str(cfg.DATASET.NAME).lower()

                if dataset_name == "voc":
                    eval_num_classes = c_num + 1
                else:
                    eval_num_classes = c_num

                print(
                    f"[Eval] dataset={dataset_name}, "
                    f"c_num={c_num}, eval_num_classes={eval_num_classes}, "
                    f"reduce_zero_label={cfg.DATASET.REDUCE_ZERO_LABEL}"
                )

                iou = mean_iou(
                    results_iou,
                    val_labels,
                    num_classes=eval_num_classes,
                    ignore_index=255,
                    nan_to_num=0,
                    reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL
                )

                print(iou["IoU"])
                if len(sfp_ratio_rows) > 0:
                    ratio_csv = os.path.join(cfg.SAVE_DIR, "urd_ratio_records.csv")

                    fieldnames = [
                        "img_idx", "filename",
                        "H", "W", "num_tokens",
                        "outliers", "ratio",
                        "score_min", "score_max", "score_mean",
                        "conf_min", "conf_max",
                        "diff_mean", "diff_max",
                        "proxy_enable",
                        "proxy_lambda",
                        "proxy_conf_thd",
                        "proxy_kernel",
                        "proxy_available_ratio",
                    ]

                    with open(ratio_csv, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=fieldnames,
                            extrasaction="ignore"
                        )
                        writer.writeheader()
                        writer.writerows(sfp_ratio_rows)

                    ratios = np.array([r["ratio"] for r in sfp_ratio_rows], dtype=np.float32)
                    outliers = np.array([r["outliers"] for r in sfp_ratio_rows], dtype=np.float32)

                    print("[URD ratio summary]")
                    print(f"num_images   = {len(ratios)}")
                    print(f"ratio mean   = {ratios.mean():.4f}")
                    print(f"ratio std    = {ratios.std():.4f}")
                    print(f"ratio min    = {ratios.min():.4f}")
                    print(f"ratio max    = {ratios.max():.4f}")
                    print(f"outlier mean = {outliers.mean():.2f}")
                    print(f"saved to     = {ratio_csv}")
                if dataset_name == "voc":
                    avg = iou["IoU"][:c_num].sum() / c_num
                else:
                    avg = iou["IoU"].sum() / c_num
                print('avg:%.4f' % (avg))
                print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, len(val_images), avg))
                print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, len(val_images), avg), file=log)
                log.write('miou:{}'.format(avg))
                if avg > best_iou:
                    best_iou = avg
                    torch.save(model.state_dict(), cfg.SAVE_DIR + 'best_weight.pth')
        if epoch == stop_epoch:
            break
    log.close()


if __name__ == '__main__':
    train(rank=0,world_size = 1)
