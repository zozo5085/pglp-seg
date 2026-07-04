# Visualization Protocol Notes

This note records the qualitative visualization rule used for PASCAL VOC
figures.

## Key Observation

Pascal VOC evaluation in this project is foreground-class mIoU. The raw VOC
labels use:

- `0`: background
- `1..20`: VOC foreground classes
- `255`: ignore

The prediction tensor contains 20 foreground channels. A plain `argmax`
therefore assigns every background pixel to one of the foreground classes. This
is numerically acceptable for foreground-only VOC mIoU after the evaluator
ignores background, but it creates noisy qualitative figures if the whole image
is overlaid.

For author-style VOC qualitative figures, draw predictions only on the VOC
foreground region:

```text
valid_visual_region = (gt_raw >= 1) & (gt_raw <= 20)
prediction[~valid_visual_region] = 255
```

This is a visualization-only step. It must not be reported as a model
improvement and must not be used for Cityscapes/ADE/Context evaluation.

## Recommended VOC Commands

Use `tools/visualize_dataset_compare.py` with `--clip-voc-gt-foreground`.

Use `--clip-voc-gt-foreground` only for qualitative comparison figures. For
debugging real background flooding, use raw masks or confidence/entropy maps
without this flag.

## Cityscapes Rule

Cityscapes has dense labels where road/building/sky/etc. are real classes.
Do not apply VOC foreground clipping. If Cityscapes visualization is poor, treat
it as a model or configuration issue:

- checkpoint/config mismatch,
- wrong text embedding or class order,
- test-time class purification pruning valid dense classes,
- overly aggressive PD/UAR thresholding,
- label-id vs train-id mapping mismatch.
