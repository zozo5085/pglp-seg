import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Source summary has 9 columns:
# 0 Image, 1 URD Score, 2 URD Mask, 3 CP Delta, 4 Before DTLR,
# 5 DTLR Delta, 6 After DTLR, 7 Attr Mask, 8 After Attr.
KEEP_COLS = [0, 1, 2, 3, 5, 8]
TITLES = [
    "Input",
    r"$S_{\mathrm{urd}}$",
    r"$M_{\mathrm{urd}}$",
    r"$\Delta_{\mathrm{CP}}$",
    r"$\Delta_{\mathrm{DTLR}}$",
    r"$L_{\mathrm{final}}$",
]
CONTENT_TOP = 38
CELL_SIZE = (230, 210)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the compact URD/DTLR diagnostic figure from a 9-column module summary image."
    )
    parser.add_argument("--input-summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def trim_white(img: Image.Image, tol: int = 248):
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < tol, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    pad = 2
    return img.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(img.size[0], int(xs.max()) + pad + 1),
            min(img.size[1], int(ys.max()) + pad + 1),
        )
    )


def fit_cover(img: Image.Image, size):
    target_w, target_h = size
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    left = max(0, (img.size[0] - target_w) // 2)
    top = max(0, (img.size[1] - target_h) // 2)
    return img.crop((left, top, left + target_w, top + target_h))


def crop_source_panels(img):
    w, h = img.size
    col_w = w / 9
    panels = []
    for idx in KEEP_COLS:
        left = int(round(idx * col_w))
        right = int(round((idx + 1) * col_w))
        panel = trim_white(img.crop((left, CONTENT_TOP, right, h))).convert("RGB")
        panels.append(np.asarray(fit_cover(panel, CELL_SIZE)))
    return panels


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panels = crop_source_panels(Image.open(args.input_summary).convert("RGB"))

    fig = plt.figure(figsize=(5.0, 3.05), dpi=int(args.dpi))
    cols, rows = 3, 2
    left, right = 0.015, 0.985
    bottom, top = 0.015, 0.955
    col_gap, row_gap = 0.012, 0.088
    ax_w = (right - left - col_gap * (cols - 1)) / cols
    ax_h = (top - bottom - row_gap * (rows - 1)) / rows

    for i, (panel, title) in enumerate(zip(panels, TITLES)):
        r, c = divmod(i, cols)
        x = left + c * (ax_w + col_gap)
        y = top - (r + 1) * ax_h - r * row_gap
        ax = fig.add_axes([x, y, ax_w, ax_h])
        ax.imshow(panel, aspect="auto")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=1.5)
        ax.axis("off")

    fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
