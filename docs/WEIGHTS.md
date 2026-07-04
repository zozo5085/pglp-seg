# Weights

Large checkpoints are not stored in git. Upload the files below to the project
cloud folder and ask users to place them in the same relative paths.

## Required Checkpoints

| Target path in repo | Dataset | Size |
| --- | --- | ---: |
| `weights/voc_pglp_seg.pth` | PASCAL VOC | 573.84 MB |
| `weights/ade_pglp_seg.pth` | ADE20K | 582.30 MB |
| `weights/city_pglp_seg.pth` | Cityscapes | 573.79 MB |

These three files are enough to run the main VOC, ADE20K, and Cityscapes
evaluation configs in this repository. They should be downloaded from the
project cloud folder and placed at the target paths above.

## Optional Text Embeddings

These files are small. They can either be uploaded with the checkpoints or
regenerated with `utils/prompt_engineering.py`.

| Target path in repo | Source file on local machine |
| --- | --- |
| `text/voc_ViT16_clip_text.pth` | `text/voc_ViT16_clip_text.pth` |
| `text/ade_ViT16_clip_text_local.pth` | `text/ade_ViT16_clip_text_local.pth` |
| `text/city_ViT16_clip_text.pth` | `text/city_ViT16_clip_text.pth` |

## Do Not Upload

Do not upload the following to the public repository or the paper artifact:

- Full epoch histories such as `checkpoint_epoch_*.pth`.
- Per-image prediction dumps under `experiments/**/**/*.pt`.
- Local visualization output folders.
- Local datasets.
- `.agents/`, `.codex-plugins/`, `.idea/`, or other local IDE/plugin folders.

## Suggested Cloud Folder Layout

```text
PGLP-Seg-Weights/
  weights/
    voc_pglp_seg.pth
    ade_pglp_seg.pth
    city_pglp_seg.pth
  text/
    voc_ViT16_clip_text.pth
    ade_ViT16_clip_text_local.pth
    city_ViT16_clip_text.pth
```
