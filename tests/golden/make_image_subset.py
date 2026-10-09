"""Copy only the images the golden runs need, so a small server does not need the full datasets.

Run it on any machine that has Cityscapes and Foggy Cityscapes, then upload ``--out``:

    python tests/golden/make_image_subset.py \
        --cityscapes /data/cityscapes/leftImg8bit \
        --foggy /data/foggy_cityscapes/leftImg8bit_foggy \
        --annotations annotations --out golden_images/cityscapes

``--cityscapes`` and ``--foggy`` must contain ``train/`` and ``val/`` with the usual ``<city>/<file>.png``.
The output keeps the layout main.py expects (the path must contain "cityscapes" so that main.py
appends the split folder):

    <out>/fog/images/val/<city>/*_foggy_beta_0.02.png      # all fog_val/<cls> images (362 for 5 classes)
    <out>/normal/images/val/<city>/*.png                  # all normal_train/<cls>/annotations_val.json images
    <out>/normal/images/train/<city>/*.png                # train images of --train_class (training smoke)
"""
import argparse
import glob
import json
import shutil
from pathlib import Path


def copy_all(names, src_root, dst_root):
    missing = 0
    for name in sorted(names):
        src, dst = src_root / name, dst_root / name
        if not src.exists():
            missing += 1
            print(f"missing: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"{dst_root}: {len(names) - missing} copied, {missing} missing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cityscapes", type=Path, required=True)
    p.add_argument("--foggy", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=Path("annotations"))
    p.add_argument("--train_class", default="bus")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    images = lambda pattern: {img["file_name"] for f in glob.glob(str(args.annotations / pattern))
                              for img in json.load(open(f))["images"]}
    copy_all(images("fog_val/*/annotations_val.json"), args.foggy / "val", args.out / "fog/images/val")
    copy_all(images("normal_train/*/annotations_val.json"), args.cityscapes / "val", args.out / "normal/images/val")
    train_file = args.annotations / "normal_train" / args.train_class / "annotations_train.jsonl"
    train = {json.loads(line)["filename"] for line in open(train_file)}
    copy_all(train, args.cityscapes / "train", args.out / "normal/images/train")


if __name__ == "__main__":
    main()
