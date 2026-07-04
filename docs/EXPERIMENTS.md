# Experiment Commands

This document records the public evaluation commands for the main reported
settings. Update dataset paths in `config/*.yaml` if your data is not stored
under `data/`.

## PASCAL VOC

Protocol: foreground-only mIoU, background ignored.

```bash
python tools/test_lrab_v1.py \
  --cfg config/voc_test_ori_cfg.yaml \
  --model PGLP_Seg \
  --model_module model.pglp_seg \
  --run-tag voc_pglp_seg
```

Expected config paths:

- checkpoint: `weights/voc_pglp_seg.pth`
- text embedding: `text/voc_ViT16_clip_text.pth`
- output: `outputs/voc_pglp_seg/`

## ADE20K

Protocol: ADE20K label space.

```bash
python tools/test_lrab_v1.py \
  --cfg config/ade_test_ori_cfg.yaml \
  --model PGLP_Seg \
  --model_module model.pglp_seg \
  --run-tag ade_pglp_seg
```

Expected config paths:

- checkpoint: `weights/ade_pglp_seg.pth`
- text embedding: `text/ade_ViT16_clip_text_local.pth`
- output: `outputs/ade_pglp_seg/`

## Cityscapes

Protocol: Cityscapes label space.

```bash
python tools/test_lrab_v1.py \
  --cfg config/cityscapes_test_ori_cfg.yaml \
  --model PGLP_Seg \
  --model_module model.pglp_seg \
  --run-tag city_pglp_seg
```

Expected config paths:

- checkpoint: `weights/city_pglp_seg.pth`
- text embedding: `text/city_ViT16_clip_text.pth`
- output: `outputs/city_pglp_seg/`

## Qualitative Figures

VOC comparison figures can be generated with:

```bash
python tools/visualize_dataset_compare.py
```

Diagnostic module figures can be generated with:

```bash
python tools/make_diagnostic_vis.py \
  --input-summary outputs/debug/module_summary.png \
  --out outputs/debug/diagnostic_module.png
```

These scripts are intended for paper visualization and assume the corresponding
prediction/debug files are already generated.
