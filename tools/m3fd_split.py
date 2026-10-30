"""Scene-level split of M3FD (RGB -> thermal extension, PLAN D2 / D3), from the outputs of ``tools/m3fd_index.py``.

M3FD has no official split, and consecutive images are often frames of the same street: a random image split puts a
near-copy of about half the test images in train. The images are in capture order (the original capture folders,
read from the xml ``path`` tag, occupy contiguous stretches of the numbering), so the split is made of scene units:
1. Cut the numbering between images i and i+1 when no image among the ``--window`` before it resembles any among the
   ``--window`` after it (64x48 thumbnail correlation below ``--vis_cut`` on Vis and ``--ir_cut`` on Ir).
2. Join the resulting stretches when they hold a near-copy (``--vis_dup`` / ``--ir_dup``, at any distance: the same
   place seen again) or images of the same numbered capture folder.
3. Assign whole units to rgb_train / ir_train / val / test so that each split gets its fraction of the images and of
   the boxes of every class (best of ``--trials`` randomised greedy fills, fixed seed).
The RGB and IR training halves share no scene (D2); val and test are used in both modalities.

Writes the split as json (image ids per split, the units, the parameters, the counts and the closest pairs across
splits); with ``--sheet``, a png of the closest cross-split pairs to check by eye.

    python tools/m3fd_split.py datasets/m3fd --out abra/data/splits/m3fd_split.json --sheet datasets/m3fd/split_check.png
"""
import argparse
import collections
import csv
import json
import re
from datetime import date
from pathlib import Path, PureWindowsPath

import numpy as np

CLASSES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
SPLITS = ("rgb_train", "ir_train", "val", "test")
NUMBERED = re.compile(r"^\d+(_.*)?$")   # capture folders are numbered ("35", "28_删", "2_warp"); relabel pools are not


def correlation(thumbs):
    x = thumbs.reshape(len(thumbs), -1).astype(np.float64)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    c = x @ x.T / x.shape[1]
    np.fill_diagonal(c, -1)
    return c


def capture_folder(path):
    d = PureWindowsPath(path).parent
    return str(d) if any(NUMBERED.match(part) for part in d.parts[1:]) else None


def units(rows, vis, ir, window, vis_cut, ir_cut, vis_dup, ir_dup):
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    cuts = 0
    for i in range(n - 1):
        before, after = slice(max(0, i - window + 1), i + 1), slice(i + 1, min(n, i + 1 + window))
        if vis[before, after].max() >= vis_cut or ir[before, after].max() >= ir_cut:
            join(i, i + 1)
        else:
            cuts += 1
    stretches = cuts + 1
    for a, b in zip(*np.nonzero(np.triu((vis > vis_dup) | (ir > ir_dup), 1))):
        join(a, b)
    first = {}
    for i, r in enumerate(rows):
        f = capture_folder(r["path"])
        if f is not None:
            join(first.setdefault(f, i), i)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values()), stretches


def assign(groups, feats, fractions, trials, seed):
    """Randomised greedy fill, largest units first: each unit goes where the squared gap between the split's share
    and its fraction (images and boxes of each class) shrinks most. Returns the best fill of all trials."""
    unit_feats = np.array([feats[g].sum(0) for g in groups])
    total, target = feats.sum(0), np.asarray(fractions)[:, None]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        sums, split = np.zeros((len(target), feats.shape[1])), np.zeros(len(groups), int)
        for u in np.argsort(-unit_feats[:, 0] * rng.uniform(0.5, 1.5, len(groups))):
            after = sums[None] + np.eye(len(target))[:, :, None] * unit_feats[u]
            k = int(np.argmin((((after / total) - target[None]) ** 2).sum((1, 2))))
            split[u] = k
            sums[k] += unit_feats[u]
        cost = float((((sums / total) - target) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, split)
    return best[1]


def closest(split_of, mats, a, b, k):
    """The k closest pairs between splits a and b: (corr, modality, image in a, image in b)."""
    ia, ib = np.nonzero(np.isin(split_of, a))[0], np.nonzero(np.isin(split_of, b))[0]
    out = []
    for name, m in mats.items():
        sub = m[np.ix_(ia, ib)]
        j = sub.argmax(1)
        for i in np.argsort(-sub.max(1))[:k]:
            out.append((float(sub[i, j[i]]), name, int(ia[i]), int(ib[j[i]])))
    return sorted(out, reverse=True)


def sheet(pairs, rows, thumbs, path):
    from PIL import Image, ImageDraw
    s, (h, w) = 3, thumbs["vis"].shape[1:]
    w, h = w * s, h * s
    canvas = Image.new("L", (4 * w + 20, len(pairs) * (h + 14)), 255)
    draw = ImageDraw.Draw(canvas)
    for r, (c, name, a, b) in enumerate(pairs):
        y = r * (h + 14)
        for col, (i, m) in enumerate(((a, "vis"), (b, "vis"), (a, "ir"), (b, "ir"))):
            canvas.paste(Image.fromarray(thumbs[m][i]).resize((w, h)), (col * w + 20 * (col >= 2), y + 12))
        draw.text((2, y), f"{name} corr {c:.3f}: {rows[a]['image']} ({rows[a]['split']}) | {rows[b]['image']} "
                          f"({rows[b]['split']}); Vis, Vis, Ir, Ir", fill=0)
    canvas.save(path)


def main(args):
    with (args.index_dir / "index.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    thumbs = np.load(args.index_dir / "thumbs.npz")
    if [r["image"] for r in rows] != thumbs["image"].tolist():
        raise SystemExit("index.csv and thumbs.npz list different images: rerun tools/m3fd_index.py")
    vis, ir = correlation(thumbs["vis"]), correlation(thumbs["ir"])
    groups, stretches = units(rows, vis, ir, args.window, args.vis_cut, args.ir_cut, args.vis_dup, args.ir_dup)
    feats = np.array([[1] + [int(r[c]) for c in CLASSES] for r in rows], dtype=float)
    unit_split = assign(groups, feats, args.fractions, args.trials, args.seed)
    split_of = np.zeros(len(rows), int)
    for g, k in zip(groups, unit_split):
        split_of[g] = k
    for r, k in zip(rows, split_of):
        r["split"] = SPLITS[k]

    counts = {s: {"units": int((unit_split == k).sum()), "images": int((split_of == k).sum()),
                  **{c: int(feats[split_of == k, j + 1].sum()) for j, c in enumerate(CLASSES)}}
              for k, s in enumerate(SPLITS)}
    mats = {"vis": vis, "ir": ir}
    checks = {"val vs train": ([2], [0, 1]), "test vs train": ([3], [0, 1]), "test vs val": ([3], [2]),
              "rgb_train vs ir_train": ([0], [1])}
    leaks = {name: closest(split_of, mats, a, b, 3) for name, (a, b) in checks.items()}
    out = {
        "dataset": "M3FD (M3FD_Detection.zip), registered Vis / Ir pairs",
        "date_created": args.date_created or date.today().isoformat(),
        "method": __doc__.split("\n\n")[1].strip(),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in ("index_dir", "out", "sheet", "date_created", "fractions")},
        "fractions": dict(zip(SPLITS, args.fractions)),
        "counts": counts,
        "closest_pairs_across_splits": {k: [{"corr": round(c, 3), "modality": m, "images": [rows[a]["image"],
                                                                                            rows[b]["image"]]}
                                            for c, m, a, b in v] for k, v in leaks.items()},
        "splits": {s: [r["image"] for r in rows if r["split"] == s] for s in SPLITS},
        "units": [[rows[i]["image"] for i in g] for g in groups],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"{stretches} stretches -> {len(groups)} units (largest {sizes[:5]}, single images {sizes.count(1)})")
    print(f"{'':10s} {'units':>5s} {'images':>6s} " + " ".join(f"{c:>10s}" for c in CLASSES))
    for s, c in counts.items():
        print(f"{s:10s} {c['units']:5d} {c['images']:6d} " + " ".join(f"{c[k]:10d}" for k in CLASSES))
    for name, v in leaks.items():
        best = {m: max(c for c, mm, _, _ in v if mm == m) for m in mats}
        print(f"closest {name}: Vis corr {best['vis']:.3f}, Ir corr {best['ir']:.3f}")
    print(f"wrote {args.out}")
    if args.sheet:
        pairs = sorted((p for v in leaks.values() for p in v), reverse=True)
        sheet(pairs[:12], rows, thumbs, args.sheet)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("index_dir", type=Path, help="folder with index.csv and thumbs.npz from tools/m3fd_index.py")
    p.add_argument("--out", type=Path, required=True, help="split json to write")
    p.add_argument("--sheet", type=Path, help="png of the closest pairs across splits")
    p.add_argument("--fractions", type=float, nargs=4, default=[0.4, 0.3, 0.1, 0.2], metavar=SPLITS,
                   help="share of the images and boxes for " + " / ".join(SPLITS))
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--vis_cut", type=float, default=0.7)
    p.add_argument("--ir_cut", type=float, default=0.9)
    p.add_argument("--vis_dup", type=float, default=0.9)
    p.add_argument("--ir_dup", type=float, default=0.95)
    p.add_argument("--trials", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--date_created", help="date written in the json (default: today)")
    args = p.parse_args()
    if abs(sum(args.fractions) - 1) > 1e-6:
        p.error("--fractions must sum to 1")
    main(args)
