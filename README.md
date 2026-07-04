# PGLP-Seg

Proxy-Guided Logit Purification for training-free CLIP-based semantic segmentation.

## Installation

```bash
conda create -n pglp-seg python=3.10 -y
conda activate pglp-seg
pip install torch torchvision
pip install ftfy regex tqdm opencv-python easydict
pip install git+https://github.com/openai/CLIP.git
```

## Dataset Layout

Place datasets under `data/`, or edit `DATAROOT` in the config files.

```text
data/
  VOC2012/
    JPEGImages/
    SegmentationClass/
    ImageSets/Segmentation/
  ADEChallengeData2016/
    images/
    annotations/
  CityScapes/
    leftImg8bit/
    gtFine/
```

## Weights

Large checkpoints are not stored in git.

Download weights from the anonymous project weight folder:

[Project weight folder](YOUR_ANONYMOUS_CLOUD_LINK)

Place files in the following paths:

```text
weights/
  voc_pglp_seg.pth
  ade_pglp_seg.pth
  city_pglp_seg.pth

text/
  voc_ViT16_clip_text.pth
  ade_ViT16_clip_text.pth
  city_ViT16_clip_text.pth
```

| Dataset | Checkpoint |
| --- | --- |
| PASCAL VOC | `weights/voc_pglp_seg.pth` |
| ADE20K | `weights/ade_pglp_seg.pth` |
| Cityscapes | `weights/city_pglp_seg.pth` |

## Evaluation

VOC:

```bash
python tools/test.py --cfg config/voc_test_ori_cfg.yaml --model PGLP_Seg --model_module model.pglp_seg
```

ADE20K:

```bash
python tools/test.py --cfg config/ade_test_ori_cfg.yaml --model PGLP_Seg --model_module model.pglp_seg
```

Cityscapes:

```bash
python tools/test.py --cfg config/cityscapes_test_ori_cfg.yaml --model PGLP_Seg --model_module model.pglp_seg
```

## Notes

- Text embeddings can be regenerated with `utils/prompt_engineering.py`.
- Pseudo-label JSON files used by training are stored in `text/`.
- Checkpoint paths use `weights/*.pth`.
- Prediction outputs are written to `outputs/`.
- Generated predictions, checkpoints, local datasets, and local visualization outputs are not tracked in git.
