"""Per-image index of M3FD (RGB -> thermal extension), to define the scene-level split (PLAN D3). The files are named
00000..04199 with no gap, so the scenes have to come from elsewhere: the xml ``folder`` / ``filename`` / ``path`` tags
(the names at capture time), the image size, and how much each image differs from the previous one. Writes
``index.csv`` (one row per image: those tags, size, boxes per class, difference to the previous Vis and Ir image) and
``thumbs.npz`` (64x48 grayscale Vis and Ir thumbnails), and prints a short summary. CPU only; reads all 8,400 PNGs,
a few minutes.

    python tools/m3fd_index.py datasets/m3fd/raw --out datasets/m3fd | tee datasets/m3fd/index_summary.txt
"""
import argparse
import collections
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image

CLASSES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
THUMB = (64, 48)


def thumb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(THUMB, Image.BOX))


def diff(a, b):
    """Mean absolute difference of two thumbnails, each standardised first (so exposure changes count little):
    0 for the same image, about 1.1 for unrelated ones."""
    a, b = ((x - x.mean()) / (x.std() + 1e-6) for x in (a.astype(np.float32), b.astype(np.float32)))
    return float(np.abs(a - b).mean())


def runs(values):
    """Runs of equal consecutive values: [value, first position, last position]."""
    out = []
    for i, v in enumerate(values):
        if out and out[-1][0] == v:
            out[-1][2] = i
        else:
            out.append([v, i, i])
    return out


def index(raw):
    anns = sorted((raw / "Annotation").glob("*.xml"))
    if not anns:
        raise SystemExit(f"no xml in {raw / 'Annotation'}")
    rows, thumbs, prev = [], {"vis": [], "ir": []}, None
    for i, a in enumerate(anns):
        r = ET.parse(a).getroot()
        objs = r.findall("object")
        names = collections.Counter(o.findtext("name") for o in objs)
        if set(names) - set(CLASSES):
            raise SystemExit(f"{a.name}: unknown class {sorted(set(names) - set(CLASSES))}")
        row = {"image": a.stem, "folder": r.findtext("folder") or "", "filename": r.findtext("filename") or "",
               "path": r.findtext("path") or "", "width": int(r.findtext("size/width")),
               "height": int(r.findtext("size/height")), "boxes": len(objs), **{c: names[c] for c in CLASSES},
               "truncated": sum(o.findtext("truncated") == "1" for o in objs)}
        cur = {m: thumb(raw / d / f"{a.stem}.png") for m, d in (("vis", "Vis"), ("ir", "Ir"))}
        for m, t in cur.items():
            row[f"{m}_diff"] = "" if prev is None else round(diff(prev[m], t), 3)
            thumbs[m].append(t)
        rows.append(row)
        prev = cur
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(anns)} images read", flush=True)
    return rows, thumbs


def summary(rows):
    def top(counter, n=12):
        return ", ".join(f"{k or '(empty)'} {v}" for k, v in counter.most_common(n)) + (" ..." if len(counter) > n else "")

    for tag, values in (("folder", [r["folder"] for r in rows]),
                        ("path dir", [str(PureWindowsPath(r["path"]).parent) for r in rows])):
        print(f"{tag}: {len(set(values))} values in {len(runs(values))} runs; {top(collections.Counter(values))}")
    odd = [r for r in rows if Path(r["filename"]).stem != r["image"]]
    shapes = collections.Counter(re.sub(r"\d+", "#", Path(r["filename"]).stem) for r in odd)
    print(f"filename != xml name: {len(odd)}; filename patterns ({len(shapes)}): {top(shapes)}")
    for r in odd[:: max(1, len(odd) // 8)][:8]:
        print(f"    {r['image']}: filename {r['filename']}, folder {r['folder']}, path {r['path']}")
    sizes = runs([f"{r['width']}x{r['height']}" for r in rows])
    other = [s for s in sizes if s[0] != "1024x768"]
    print(f"size: {len(sizes)} runs; not 1024x768: "
          + ", ".join(f"{rows[a]['image']}-{rows[b]['image']} {v}" for v, a, b in other[:30])
          + (" ..." if len(other) > 30 else ""))
    for m in ("vis", "ir"):
        d = np.array([r[f"{m}_diff"] for r in rows[1:]], dtype=float)
        q = np.percentile(d, [50, 90, 95, 99])
        print(f"{m}_diff to the previous image: median {q[0]:.2f}, p90 {q[1]:.2f}, p95 {q[2]:.2f}, p99 {q[3]:.2f}, "
              f"max {d.max():.2f}; above 0.6 / 0.8 / 1.0: {(d > .6).sum()} / {(d > .8).sum()} / {(d > 1).sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("raw", type=Path, help="unpacked M3FD_Detection.zip (holds Annotation/, Ir/, Vis/)")
    p.add_argument("--out", type=Path, required=True, help="folder for index.csv and thumbs.npz")
    args = p.parse_args()
    rows, thumbs = index(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(args.out / "thumbs.npz", image=np.array([r["image"] for r in rows]),
                        vis=np.stack(thumbs["vis"]), ir=np.stack(thumbs["ir"]))
    print(f"wrote {args.out / 'index.csv'} and {args.out / 'thumbs.npz'}")
    summary(rows)
