"""Find the test images behind the supplementary qualitative panels (revision round 2, T3), CPU only.

The panels (``supplementary/images/<class>_gt.png``, 500 x 250) are whole Foggy Cityscapes test images resized, with
the ground-truth boxes of the class drawn in #FFC600. Every image of the class's per-class test file is resized to
the panel size and compared with the panel outside the drawn boxes (mean absolute difference); the closest one is
the image. For each class this writes ``<out>/<class>/annotations_val.json`` (that image and its boxes of the class,
the per-class test file restricted to it), a copy of the image, and ``<out>/match.json`` (best and runner-up).

    python tools/qual_find.py --panels <dir with <class>_gt.png> --images datasets/cityscapes/fog/images/val \
        --annotations datasets/cityscapes/fog/annotations/experts --classes bus rider train --out qual
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BOX = np.array([0xFF, 0xC6, 0x00])   # ground-truth box colour of the panels


def mask_outside_boxes(panel):
    """True where the panel shows the image (not a drawn box), dilated by 2 px around box pixels."""
    near = (np.abs(panel.astype(int) - BOX).sum(-1) < 120)
    grow = near.copy()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            grow |= np.roll(np.roll(near, dy, 0), dx, 1)
    return ~grow


def match(panel_path, images_root, test):
    panel = np.asarray(Image.open(panel_path).convert("RGB"))
    keep = mask_outside_boxes(panel)
    scores = []
    for img in test["images"]:
        im = Image.open(images_root / img["file_name"]).convert("RGB").resize(panel.shape[1::-1], Image.BILINEAR)
        diff = np.abs(np.asarray(im).astype(int) - panel.astype(int)).sum(-1)
        scores.append((float(diff[keep].mean()), img))
    scores.sort(key=lambda s: s[0])
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="test image folder (file names of the test json)")
    p.add_argument("--annotations", type=Path, required=True, help="folder of <class>/annotations_val.json")
    p.add_argument("--classes", nargs="+", default=["bus", "rider", "train"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    report = {}
    for cls in args.classes:
        test = json.loads((args.annotations / cls / "annotations_val.json").read_text())
        scores = match(args.panels / f"{cls}_gt.png", args.images, test)
        (best, img), (second, img2) = scores[0], scores[1]
        report[cls] = {"file_name": img["file_name"], "image_id": img["id"], "mean_abs_diff": best,
                       "runner_up": img2["file_name"], "runner_up_diff": second, "candidates": len(scores)}
        one = {**test, "images": [img], "annotations": [a for a in test["annotations"] if a["image_id"] == img["id"]]}
        out = args.out / cls
        out.mkdir(parents=True, exist_ok=True)
        (out / "annotations_val.json").write_text(json.dumps(one))
        shutil.copy(args.images / img["file_name"], out / "image.png")
        print(f"{cls}: {img['file_name']} (diff {best:.2f}; runner-up {second:.2f}), {len(one['annotations'])} boxes")
    (args.out / "match.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
