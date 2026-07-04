# PGLP-Seg

Proxy-Guided Logit Purification for training-free CLIP-based semantic
segmentation.

This repository contains the implementation, configuration files, and evaluation
utilities used in the paper. The method is implemented as an inference-time
refinement module on top of a frozen CLIP-based dense prediction pipeline.

## What Is Included

- `model/pglp_seg.py`: public PGLP-Seg model entrypoint.
- `model/model.py`: implementation module.
- `tools/test.py`: evaluation script with per-class/per-image metrics
  and optional diagnostic map export.
- `tools/train.py`: rectification-stage training script.
- `tools/visualize_dataset_compare.py`: qualitative comparison utility.
- `tools/make_diagnostic_vis.py`: diagnostic module figure helper.
- `config/*_ori_cfg.yaml`: dataset-specific experiment settings.
- `docs/WEIGHTS.md`: required checkpoint list and expected placement.
- `docs/EXPERIMENTS.md`: commands used for the reported experiments.

Generated predictions, checkpoints, local datasets, and local visualization
outputs are not tracked in git.

## Installation

```bash
conda create -n pglp-seg python=3.10 -y
conda activate pglp-seg
pip install torch torchvision
pip install ftfy regex tqdm opencv-python easydict
pip install git+https://github.com/openai/CLIP.git
```

Use a PyTorch/CUDA build matching your GPU driver. The reported experiments were
run with CUDA-enabled PyTorch.

## Dataset Layout

Place datasets under `data/` or edit `DATAROOT` in the config files.

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

Large checkpoints are distributed separately. Download them from the project
cloud link and place them under `weights/`.

| Dataset | Download | Save as |
| --- | --- | --- |
| PASCAL VOC | [86.20 mIoU](https://drive.google.com/drive/folders/1ZKs5_LCFU_QBRXABJTm-MPOJOvIorqal) | `weights/voc_pglp_seg.pth` |
| ADE20K | [18.02 mIoU](https://drive.google.com/drive/folders/1ZKs5_LCFU_QBRXABJTm-MPOJOvIorqal) | `weights/ade_pglp_seg.pth` |
| Cityscapes | [40.44 mIoU](https://drive.google.com/drive/folders/1ZKs5_LCFU_QBRXABJTm-MPOJOvIorqal) | `weights/city_pglp_seg.pth` |

Text embeddings are included under `text/`. See `docs/WEIGHTS.md` for details.

## Evaluation

VOC:

```bash
python tools/test.py \
  --cfg config/voc_test_ori_cfg.yaml \
  --model PGLP_Seg \
  --model_module model.pglp_seg
```

ADE20K:

```bash
python tools/test.py \
  --cfg config/ade_test_ori_cfg.yaml \
  --model PGLP_Seg \
  --model_module model.pglp_seg
```

Cityscapes:

```bash
python tools/test.py \
  --cfg config/cityscapes_test_ori_cfg.yaml \
  --model PGLP_Seg \
  --model_module model.pglp_seg
```

PASCAL VOC follows the foreground-only protocol used by the baseline evaluation.
ADE20K and Cityscapes are evaluated under their corresponding dataset label
spaces.

## Reproducibility Notes

- Text embeddings can be generated with `utils/prompt_engineering.py`.
- Pseudo-label JSON files used for training are stored in `text/`.
- Checkpoint paths in configs use `weights/*.pth`.
- Prediction outputs are written to `outputs/`.

## Acknowledgement

This code builds on CLIP-based dense prediction. Please also cite the
corresponding baseline work when using this repository.
