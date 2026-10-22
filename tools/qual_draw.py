"""Supplementary qualitative panels (revision round 2, T3) from the outputs of ``revision2.sh qual``.

    python tools/qual_draw.py --qual qual --out supplementary/images [--style pretty|original]

Input per class: ``<qual>/<class>/{image.png, annotations_val.json, pred_source.json, pred_abra.json}`` (predictions
of the evaluated class with score >= 0.35, COCO xywh). Output: ``<class>_{source,abra,gt}.png`` (2:1, like the
originals) and ``report.json`` (per image: ground-truth boxes found by each model at IoU >= 0.5, the best IoU).

Styles: ``original`` = 500 x 250, prediction boxes #7CFF00, ground truth #FFC600, 4 px (the submission's panels;
their GT panels can be kept). ``pretty`` = 1000 x 500, anti-aliased boxes, and a zoom inset of the same region in the
three panels of a row (the ground-truth objects, magnified), placed in the bottom corner away from them.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

PRED, GT, FRAME = "#7CFF00", "#FFC600", "#FFFFFF"


def boxes(path):
    return [a["bbox"] for a in json.loads(Path(path).read_text())["annotations"]]


def iou(a, b):   # COCO xywh
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - w * h
    return w * h / union if union else 0.0


def draw(image, rects, color, width, region=None, size=None, ss=3):
    """``image`` cropped to ``region`` (x0, y0, x1, y1; default whole), resized to ``size``, ``rects`` (xywh in image
    pixels) drawn with ``width`` px lines; anti-aliased by drawing at ``ss`` x the size and downsampling."""
    region = region or (0, 0, image.width, image.height)
    big = image.crop(region).resize((size[0] * ss, size[1] * ss), Image.LANCZOS)
    sx, sy = big.width / (region[2] - region[0]), big.height / (region[3] - region[1])
    pen = ImageDraw.Draw(big)
    for x, y, w, h in rects:
        pen.rectangle([(x - region[0]) * sx, (y - region[1]) * sy, (x + w - region[0]) * sx, (y + h - region[1]) * sy],
                      outline=color, width=width * ss)
    return big.resize(size, Image.LANCZOS)


def zoom_region(gt, image, min_size=(256, 128), factor=2.5):
    """2:1 region around the ground-truth boxes: ``factor`` x their extent, at least ``min_size``, inside the image."""
    x0, y0 = min(b[0] for b in gt), min(b[1] for b in gt)
    x1, y1 = max(b[0] + b[2] for b in gt), max(b[1] + b[3] for b in gt)
    w = max(min_size[0], factor * (x1 - x0), 2 * factor * (y1 - y0))
    w = min(w, image.width / 2)
    h = w / 2
    cx = min(max((x0 + x1) / 2, w / 2), image.width - w / 2)
    cy = min(max((y0 + y1) / 2, h / 2), image.height - h / 2)
    return round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)


def pretty(image, rects, color, gt, size=(1000, 500), inset_frac=0.42, pad=10):
    panel = draw(image, rects, color, 3, size=size)
    region = zoom_region(gt, image)
    iw, ih = round(size[0] * inset_frac), round(size[1] * inset_frac)
    inset = draw(image, rects, color, 3, region=region, size=(iw, ih))
    s = size[0] / image.width
    right = (region[0] + region[2]) / 2 < image.width / 2   # inset on the side away from the objects
    ox, oy = (size[0] - iw - pad, size[1] - ih - pad) if right else (pad, size[1] - ih - pad)
    pen = ImageDraw.Draw(panel)
    pen.rectangle([region[0] * s, region[1] * s, region[2] * s, region[3] * s], outline=FRAME, width=2)
    pen.rectangle([ox - 3, oy - 3, ox + iw + 2, oy + ih + 2], fill=FRAME)
    panel.paste(inset, (ox, oy))
    return panel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qual", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--classes", nargs="+", default=["bus", "rider", "train"])
    p.add_argument("--style", choices=["pretty", "original"], default="pretty")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for cls in args.classes:
        d = args.qual / cls
        image = Image.open(d / "image.png").convert("RGB")
        gt = boxes(d / "annotations_val.json")
        panels = {"gt": (gt, GT)} | {m: (boxes(d / f"pred_{m}.json"), PRED) for m in ("source", "abra")}
        for name, (rects, color) in panels.items():
            if args.style == "pretty":
                img = pretty(image, rects, color, gt)
            else:
                img = draw(image, rects, color, 4, size=(500, 250), ss=1)
            img.save(args.out / f"{cls}_{name}.png", optimize=True)
        report[cls] = {m: {"predictions": len(panels[m][0]),
                           "gt_found_iou50": sum(any(iou(g, q) >= 0.5 for q in panels[m][0]) for g in gt),
                           "best_iou_per_gt": [round(max((iou(g, q) for q in panels[m][0]), default=0.0), 3) for g in gt]}
                       for m in ("source", "abra")} | {"gt_boxes": len(gt)}
        print(cls, report[cls])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
