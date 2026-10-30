"""Crops of one M3FD class, Vis next to Ir, to choose its prompt (PLAN §3: is "Lamp" a street lamp or a traffic
light?). Boxes are drawn at random (fixed seed, at most one per image), with some context around them and the box
outlined. CPU, PIL only.

    python tools/m3fd_crops.py datasets/m3fd/raw --cls Lamp --out datasets/m3fd/lamp_crops.png
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


def boxes(raw, cls):
    out = []
    for f in sorted((raw / "Annotation").glob("*.xml")):
        found = [[float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
                 for o in ET.parse(f).getroot().findall("object") if o.findtext("name") == cls]
        out += [(f.stem, b) for b in found]
    return out


def tile(path, box, context, size):
    """Crop around the box (context = margin as a share of the box's larger side), padded to a square tile."""
    x1, y1, x2, y2 = box
    m = context * max(x2 - x1, y2 - y1, 8)
    with Image.open(path) as im:
        im = im.convert("RGB")
        c = (max(0, int(x1 - m)), max(0, int(y1 - m)), min(im.width, int(x2 + m) + 1), min(im.height, int(y2 + m) + 1))
        crop = im.crop(c)
    ImageDraw.Draw(crop).rectangle([x1 - c[0], y1 - c[1], x2 - c[0], y2 - c[1]], outline=(255, 40, 40))
    scale = size / max(crop.size)   # small boxes are enlarged too
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BICUBIC)
    out = Image.new("RGB", (size, size), (255, 255, 255))
    out.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return out


def main(args):
    found = boxes(args.raw, args.cls)
    if not found:
        raise SystemExit(f"no {args.cls} box in {args.raw / 'Annotation'}")
    rng = random.Random(args.seed)
    per_image = {}
    for image, b in rng.sample(found, len(found)):
        per_image.setdefault(image, b)
    picked = sorted(rng.sample(sorted(per_image.items()), min(args.n, len(per_image))))
    s, cols = args.tile, args.cols
    sheet = Image.new("RGB", (cols * (2 * s + 12), -(-len(picked) // cols) * (s + 16)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (image, b) in enumerate(picked):
        x, y = (k % cols) * (2 * s + 12), (k // cols) * (s + 16)
        for j, folder in enumerate(("Vis", "Ir")):
            sheet.paste(tile(args.raw / folder / f"{image}.png", b, args.context, s), (x + j * s, y + 14))
        draw.text((x, y), f"{image}  {b[2] - b[0]:.0f}x{b[3] - b[1]:.0f} px", fill=(0, 0, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"{len(found)} {args.cls} boxes in {len({i for i, _ in found})} images; {len(picked)} drawn -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("raw", type=Path, help="unpacked M3FD_Detection.zip (holds Annotation/, Ir/, Vis/)")
    p.add_argument("--cls", default="Lamp", help="M3FD class name as in the xml")
    p.add_argument("--out", type=Path, required=True, help="png to write")
    p.add_argument("--n", type=int, default=48, help="boxes to draw")
    p.add_argument("--context", type=float, default=1.0, help="margin around the box, as a share of its larger side")
    p.add_argument("--tile", type=int, default=128, help="tile size in px (one for Vis, one for Ir)")
    p.add_argument("--cols", type=int, default=6, help="Vis / Ir pairs per row")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
