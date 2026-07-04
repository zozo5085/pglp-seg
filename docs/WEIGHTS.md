# Weights

Large checkpoints are not stored in git. Upload the files below to the project
cloud folder and ask users to place them in the same relative paths.

## Required Checkpoints

| Target path in repo | Dataset | Download / reported score | Size |
| --- | --- | --- | ---: |
| `weights/voc_pglp_seg.pth` | PASCAL VOC | [86.20 mIoU](https://drive.google.com/file/d/1B53lYGwJIlDEhRQ5fob7UkmixrvUh7kj/view) | 573.84 MB |
| `weights/ade_pglp_seg.pth` | ADE20K | [18.02 mIoU](https://drive.google.com/file/d/1P-ab36_-nU-Mcntnb8PRKXFTaWsq6FGR/view?usp=drive_link) | 582.30 MB |
| `weights/city_pglp_seg.pth` | Cityscapes | [40.44 mIoU](https://drive.google.com/file/d/1dzcEQIhla8cIKOKTEw5mMWzFwvlbpiJf/view?usp=drive_link) | 573.79 MB |

These three files are enough to run the main VOC, ADE20K, and Cityscapes
evaluation configs in this repository. They should be downloaded from the
project cloud folder and placed at the target paths above.

## Optional Text Embeddings

These files are small and are included in this repository. They can also be
regenerated with `utils/prompt_engineering.py`.

| Target path in repo | Status |
| --- | --- |
| `text/voc_ViT16_clip_text.pth` | included |
| `text/ade_ViT16_clip_text.pth` | included |
| `text/city_ViT16_clip_text.pth` | included |

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
    ade_ViT16_clip_text.pth
    city_ViT16_clip_text.pth
```
