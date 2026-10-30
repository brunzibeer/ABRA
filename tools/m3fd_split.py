"""Scene-level split of M3FD (RGB -> thermal extension, PLAN D2 / D3), from the outputs of ``tools/m3fd_index.py``.

M3FD has no official split, and consecutive images are often frames of the same street: a random image split puts a
near-copy of about half the test images in train. The images are in capture order (the original capture folders,
read from the xml ``path`` tag, occupy contiguous stretches of the numbering), so the split is made of blocks of that
order. Thumbnail similarity alone does not find scenes (frames of one capture folder go down to 0.3 correlation, and
one drive continues across folders), so blocks are cut only at clear scene changes:
1. Never cut inside a numbered capture folder (from its first to its last image) or inside a run of images of one
   size other than 1024x768 (one camera setup).
2. Elsewhere, cut between images i and i+1 when no image among the ``--window`` before it resembles any among the
   ``--window`` after it: correlation of thumbnails pooled to 16x12 (``--pool``, tolerant to moving people and cars)
   below ``--vis_cut`` on Vis and below ``--ir_cut`` on Ir.
3. Keep a cut only if the ``--verify_window`` frames on each side are different places also when the camera moved:
   best correlation over translations and zooms (``Shifted``) below ``--same_either`` in both modalities and not above
   ``--same_both`` in both. Then cuts that leave a block shorter than ``--min_block`` are noise inside one scene (dark
   night frames, mostly) and are dropped on both sides.
4. Join blocks that hold the same place seen again, at any distance: a near-identical image in one modality
   (``--vis_dup`` / ``--ir_dup``, also catches a duplicated pair), or at least ``--revisit_pairs`` pairs between the
   two blocks with a shift-tolerant correlation of ``--revisit_same`` in both modalities (candidates: coarse
   correlation above ``--revisit_vis`` and ``--revisit_ir``). One such pair alone is common between unrelated roads.
5. Assign whole units to rgb_train / ir_train / val / test so that each split gets its fraction of the images and of
   the boxes of every class (randomised greedy fills improved by moves and swaps of units, best of ``--trials``,
   fixed seed), with floors on the boxes of the classes to transport (PLAN D5: ``--d5_rgb_train`` in rgb_train,
   ``--d5_test`` in test, for ``--d5_classes``). ``--balance`` adds image conditions to balance as well: ``light``
   (night: mean Vis thumbnail below ``--dark_below``), ``haze`` (bright but low contrast: std below
   ``--haze_std_below``), ``size`` (not 1024x768).
6. Leave out the ``--guard`` images on each side of every boundary between two splits: thumbnails miss some scene
   continuations (a camera turning, people walking past), and these frames are where they would leak. ``--drop``
   leaves out broken pairs (01300: its Vis is a copy of 00536 and its Ir of 00539).
The RGB and IR training halves share no scene (D2); val and test are used in both modalities.

Writes the split as json (image ids per split, the units, the parameters, the counts, every block boundary between two
splits and the most similar pairs across splits). ``--sheet`` draws both as a png, to check by eye. The two versions
kept (PLAN §3):

    python tools/m3fd_split.py datasets/m3fd --out abra/data/splits/m3fd_split_v1.json --date_created 2026-10-30
    python tools/m3fd_split.py datasets/m3fd --out abra/data/splits/m3fd_split_v2.json --date_created 2026-10-30 \\
        --balance light haze size
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


def correlation(thumbs, pool=1):
    """Correlation between all thumbnails, after average pooling by ``pool`` (coarser = tolerant to moving objects)."""
    n, h, w = thumbs.shape
    x = thumbs.astype(np.float64).reshape(n, h // pool, pool, w // pool, pool).mean((2, 4)).reshape(n, -1)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    c = x @ x.T / x.shape[1]
    np.fill_diagonal(c, -1)
    return c


def capture_folder(path):
    d = PureWindowsPath(path).parent
    return str(d) if any(NUMBERED.match(part) for part in d.parts[1:]) else None


class Shifted:
    """Best correlation of two thumbnails over translations (up to ``max_shift`` of the image) and a few zooms (the car
    drove forward), after pooling by 2: tolerant to the camera moving between two frames of one scene."""

    def __init__(self, thumbs, max_shift=0.25, zooms=(1.15, 1.3)):
        from PIL import Image
        n, h, w = thumbs.shape

        def pooled(x):
            return x.astype(np.float32).reshape(len(x), h // 2, 2, w // 2, 2).mean((2, 4))

        def zoom(t, z):
            ch, cw = round(h / z), round(w / z)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            return np.asarray(Image.fromarray(t[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.BILINEAR))

        self.base = pooled(thumbs)
        self.zoomed = [pooled(np.stack([zoom(t, z) for t in thumbs])) for z in zooms]
        hh, ww = self.base.shape[1:]
        my, mx = int(hh * max_shift), int(ww * max_shift)
        self.shifts = [(dy, dx) for dy in range(-my, my + 1, 2) for dx in range(-mx, mx + 1, 2)]

    def __call__(self, pairs, chunk=4096):
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        best = np.full(len(pairs), -1.0)
        for c in range(0, len(pairs), chunk):
            a, b = pairs[c:c + chunk, 0], pairs[c:c + chunk, 1]
            combos = [(self.base[a], self.base[b])]
            combos += [(self.base[a], z[b]) for z in self.zoomed] + [(self.base[b], z[a]) for z in self.zoomed]
            for x_all, y_all in combos:
                hh, ww = x_all.shape[1:]
                for dy, dx in self.shifts:
                    x = x_all[:, max(0, dy):hh + min(0, dy), max(0, dx):ww + min(0, dx)].reshape(len(a), -1)
                    y = y_all[:, max(0, -dy):hh + min(0, -dy), max(0, -dx):ww + min(0, -dx)].reshape(len(a), -1)
                    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
                    y = (y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-6)
                    np.maximum(best[c:c + chunk], (x * y).mean(1), out=best[c:c + chunk])
        return best


def units(rows, thumbs, vis, ir, args):
    """-> (units as sorted lists of positions, positions i where the order is cut between i and i + 1, number of
    revisits joined). ``vis`` / ``ir``: correlation of the 64x48 thumbnails."""
    n = len(rows)
    locked = np.zeros(n - 1, bool)
    spans = {}
    for i, r in enumerate(rows):
        f = capture_folder(r["path"])
        if f is not None:
            spans[f] = (spans.get(f, (i, i))[0], i)
    for a, b in spans.values():
        locked[a:b] = True
    size = [(r["width"], r["height"]) for r in rows]
    locked |= np.array([size[i] == size[i + 1] != ("1024", "768") for i in range(n - 1)])
    coarse_vis, coarse_ir = correlation(thumbs["vis"], args.pool), correlation(thumbs["ir"], args.pool)
    cuts = []
    for i in np.nonzero(~locked)[0]:
        before, after = slice(max(0, i - args.window + 1), i + 1), slice(i + 1, min(n, i + 1 + args.window))
        if coarse_vis[before, after].max() < args.vis_cut and coarse_ir[before, after].max() < args.ir_cut:
            cuts.append(int(i))

    # a cut is kept only if the frames next to it are different places also when the camera may have moved
    shifted = {m: Shifted(thumbs[m]) for m in ("vis", "ir")}
    w = args.verify_window
    near = [(c, a, b) for c in cuts for a in range(max(0, c - w + 1), c + 1) for b in range(c + 1, min(n, c + 1 + w))]
    if near:
        sv, si = (shifted[m]([(a, b) for _, a, b in near]) for m in ("vis", "ir"))
        same = (np.maximum(sv, si) >= args.same_either) | (np.minimum(sv, si) >= args.same_both)
        cuts = sorted(set(cuts) - {c for (c, _, _), s in zip(near, same) if s})
    # cuts a few images apart are noise inside one scene, not two scene changes: drop both cuts of a short block
    while cuts:
        edges = [-1] + cuts + [n - 1]
        k = next((k for k in range(1, len(edges)) if edges[k] - edges[k - 1] < args.min_block), None)
        if k is None:
            break
        cuts = [c for c in cuts if c not in (edges[k - 1], edges[k])]
    block = np.zeros(n, int)
    for i in cuts:
        block[i + 1:] += 1
    parent = list(range(len(cuts) + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # the same place seen again, at any distance: a near-identical image in one modality, or both modalities alike
    links = list(zip(*np.nonzero(np.triu((vis > args.vis_dup) | (ir > args.ir_dup), 1))))
    other = block[:, None] != block[None, :]
    cand = np.argwhere(np.triu(other & (coarse_vis > args.revisit_vis) & (coarse_ir > args.revisit_ir), 1))
    if len(cand):   # one look-alike pair is common between unrelated roads; a revisit gives several
        sv, si = shifted["vis"](cand), shifted["ir"](cand)
        alike = collections.defaultdict(list)
        for a, b in cand[np.minimum(sv, si) >= args.revisit_same]:
            alike[tuple(sorted((block[a], block[b])))].append((a, b))
        links += [pairs[0] for pairs in alike.values() if len(pairs) >= args.revisit_pairs]
    joined = 0
    for a, b in links:
        ra, rb = find(block[a]), find(block[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            joined += 1
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(block[i])].append(i)
    return sorted(groups.values()), cuts, joined


def assign(groups, feats, fractions, trials, seed, floors=(), weights=None):
    """Randomised greedy fill, largest units first: each unit goes where the squared gap between the split's share
    and its fraction (images, boxes of each class, balanced conditions) shrinks most. Then a local search moves one
    unit to another split, or swaps two units, while that lowers the gap. ``floors``: (split, column, minimum) that a
    fill should reach (D5 box counts), as a steep penalty. Returns the best result of all trials."""
    unit_feats = np.array([feats[g].sum(0) for g in groups])
    total, target = feats.sum(0), np.asarray(fractions)[:, None]
    n_splits, n_units = len(target), len(groups)
    weights = np.ones(feats.shape[1]) if weights is None else np.asarray(weights, float)

    def cost(sums):
        short = sum(max(0.0, 1 - sums[k, c] / m) ** 2 for k, c, m in floors)
        return float(((((sums / total) - target) ** 2) * weights).sum()) + 100 * short

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        sums, split = np.zeros((n_splits, feats.shape[1])), np.zeros(n_units, int)
        for u in np.argsort(-unit_feats[:, 0] * rng.uniform(0.5, 1.5, n_units)):
            k = int(np.argmin([cost(sums + np.eye(n_splits)[j][:, None] * unit_feats[u]) for j in range(n_splits)]))
            split[u] = k
            sums[k] += unit_feats[u]
        current, improved = cost(sums), True
        while improved:
            improved = False
            for u in range(n_units):
                for k in range(n_splits):
                    if k == split[u]:
                        continue
                    trial = sums.copy()
                    trial[split[u]] -= unit_feats[u]
                    trial[k] += unit_feats[u]
                    if cost(trial) < current - 1e-12:
                        sums, split[u], current, improved = trial, k, cost(trial), True
            for u in range(n_units):
                for v in range(u + 1, n_units):
                    if split[u] == split[v]:
                        continue
                    trial = sums.copy()
                    trial[split[u]] += unit_feats[v] - unit_feats[u]
                    trial[split[v]] += unit_feats[u] - unit_feats[v]
                    if cost(trial) < current - 1e-12:
                        sums, current, improved = trial, cost(trial), True
                        split[u], split[v] = split[v], split[u]
        if best is None or current < best[0]:
            best = (current, split.copy())
    return best[1]


def closest(split_of, vis, ir, a, b, k):
    """The k most similar pairs between splits a and b, by Vis + Ir correlation: (vis, ir, image in a, image in b)."""
    ia, ib = np.nonzero(np.isin(split_of, a))[0], np.nonzero(np.isin(split_of, b))[0]
    score = vis[np.ix_(ia, ib)] + ir[np.ix_(ia, ib)]
    top = np.argsort(-score, axis=None)[:k]
    return [(float(vis[ia[i], ib[j]]), float(ir[ia[i], ib[j]]), int(ia[i]), int(ib[j]))
            for i, j in zip(*np.unravel_index(top, score.shape))]


def sheet(cells, rows, thumbs, path, per_row=3):
    """cells: (label, image a, image b), drawn as Vis a, Vis b, Ir a, Ir b."""
    from PIL import Image, ImageDraw
    h, w = (x * 3 // 2 for x in thumbs["vis"].shape[1:])
    cw, ch = 4 * w + 10, h + 16
    canvas = Image.new("L", (per_row * (cw + 16), -(-len(cells) // per_row) * ch), 255)
    draw = ImageDraw.Draw(canvas)
    for k, (label, a, b) in enumerate(cells):
        x, y = (k % per_row) * (cw + 16), (k // per_row) * ch
        for col, (i, m) in enumerate(((a, "vis"), (b, "vis"), (a, "ir"), (b, "ir"))):
            canvas.paste(Image.fromarray(thumbs[m][i]).resize((w, h)), (x + col * w + 10 * (col >= 2), y + 14))
        draw.text((x, y), f"{label}: {rows[a]['image']} {rows[a]['assigned']} | {rows[b]['image']} {rows[b]['assigned']}",
                  fill=0)
    canvas.save(path)


def conditions(rows, thumbs, dark_below, haze_std_below):
    """Per image, from the Vis thumbnail and the xml size: night (dark), haze (bright, low contrast), other size."""
    v = thumbs["vis"].astype(np.float64)
    mean, std = v.mean((1, 2)), v.std((1, 2))
    return {"light": mean < dark_below, "haze": (mean >= dark_below) & (std < haze_std_below),
            "size": np.array([(r["width"], r["height"]) != ("1024", "768") for r in rows])}


def main(args):
    with (args.index_dir / "index.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    thumbs = np.load(args.index_dir / "thumbs.npz")
    if [r["image"] for r in rows] != thumbs["image"].tolist():
        raise SystemExit("index.csv and thumbs.npz list different images: rerun tools/m3fd_index.py")
    vis, ir = correlation(thumbs["vis"]), correlation(thumbs["ir"])
    drop = [i for i, r in enumerate(rows) if r["image"] in args.drop]
    for i in drop:   # a broken pair (Vis and Ir copied from two other frames) would join unrelated blocks
        vis[i, :] = vis[:, i] = ir[i, :] = ir[:, i] = -1
    groups, cuts, joined = units(rows, thumbs, vis, ir, args)
    cond = conditions(rows, thumbs, args.dark_below, args.haze_std_below)
    feats = np.array([[1] + [int(r[c]) for c in CLASSES] for r in rows], dtype=float)
    column = {c: j + 1 for j, c in enumerate(CLASSES)}
    floors = [(SPLITS.index(s), column[c], m) for c in args.d5_classes
              for s, m in (("rgb_train", args.d5_rgb_train), ("test", args.d5_test))]
    unit_split = assign(groups, np.column_stack([feats] + [cond[b] for b in args.balance]), args.fractions,
                        args.trials, args.seed, floors, [1] * feats.shape[1] + [args.balance_weight] * len(args.balance))
    split_of = np.zeros(len(rows), int)
    for g, k in zip(groups, unit_split):
        split_of[g] = k
    # frames next to a boundary between two splits may still show the same scene: they go to no split
    boundaries = [i for i in cuts if split_of[i] != split_of[i + 1]]
    guard = sorted({j for i in boundaries for j in range(i - args.guard + 1, i + args.guard + 1) if 0 <= j < len(rows)})
    kept = np.ones(len(rows), bool)
    kept[guard + drop] = False
    for r, k, keep in zip(rows, split_of, kept):
        r["assigned"], r["split"] = SPLITS[k], (SPLITS[k] if keep else "excluded")

    names = {"light": "night", "haze": "haze", "size": "not_1024x768"}
    counts = {s: {"units": int((unit_split == k).sum()), "images": int(((split_of == k) & kept).sum()),
                  **{c: int(feats[(split_of == k) & kept, j + 1].sum()) for j, c in enumerate(CLASSES)},
                  **{names[b]: int(cond[b][(split_of == k) & kept].sum()) for b in names}}
              for k, s in enumerate(SPLITS)}
    checks = {"val vs train": ([2], [0, 1]), "test vs train": ([3], [0, 1]), "test vs val": ([3], [2]),
              "rgb_train vs ir_train": ([0], [1])}
    similar = {name: closest(np.where(kept, split_of, -1), vis, ir, a, b, 5) for name, (a, b) in checks.items()}
    out = {
        "dataset": "M3FD (M3FD_Detection.zip), registered Vis / Ir pairs",
        "date_created": args.date_created or date.today().isoformat(),
        "method": __doc__.split("\n\n")[1].strip(),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in ("index_dir", "out", "sheet", "date_created", "fractions")},
        "fractions": dict(zip(SPLITS, args.fractions)),
        "counts": counts,
        "boundaries_between_splits": [[rows[i]["image"], rows[i + 1]["image"]] for i in boundaries],
        "excluded_by_guard": [rows[i]["image"] for i in guard],
        "dropped": [rows[i]["image"] for i in drop],
        "most_similar_pairs_across_splits": {k: [{"vis": round(v, 3), "ir": round(r, 3),
                                                  "images": [rows[a]["image"], rows[b]["image"]]}
                                                 for v, r, a, b in pairs] for k, pairs in similar.items()},
        "splits": {s: [r["image"] for r in rows if r["split"] == s] for s in SPLITS},
        "units": [[rows[i]["image"] for i in g] for g in groups],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"{len(cuts) + 1} blocks, {joined} joined as the same place -> {len(groups)} units (largest {sizes[:5]}); "
          f"{len(boundaries)} boundaries between splits, {len(guard)} images next to them excluded")
    print(f"balanced on: classes{''.join(' + ' + b for b in args.balance)}")
    for c in args.d5_classes:
        if counts["rgb_train"][c] < args.d5_rgb_train or counts["test"][c] < args.d5_test:
            print(f"WARNING: {c} below the D5 floor: rgb_train {counts['rgb_train'][c]} (>= {args.d5_rgb_train}), "
                  f"test {counts['test'][c]} (>= {args.d5_test})")
    print(f"{'':10s} {'units':>5s} {'images':>6s} " + " ".join(f"{c:>10s}" for c in CLASSES) + "  night  haze  size")
    for s, c in counts.items():
        print(f"{s:10s} {c['units']:5d} {c['images']:6d} " + " ".join(f"{c[k]:10d}" for k in CLASSES) + "  "
              + "  ".join(f"{c[names[b]] / c['images']:4.0%}" for b in names))
    for name, pairs in similar.items():
        v, r, a, b = pairs[0]
        print(f"most similar {name}: {rows[a]['image']} / {rows[b]['image']} (Vis {v:.3f}, Ir {r:.3f})")
    print(f"wrote {args.out}")
    if args.sheet:
        cells = [(f"boundary {k + 1}", i, i + 1) for k, i in enumerate(boundaries)]
        cells += [(f"similar, Vis {v:.2f} Ir {r:.2f}", a, b) for pairs in similar.values() for v, r, a, b in pairs]
        sheet(cells, rows, thumbs, args.sheet)
        print(f"wrote {args.sheet}: {len(boundaries)} boundaries, then the most similar pairs")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("index_dir", type=Path, help="folder with index.csv and thumbs.npz from tools/m3fd_index.py")
    p.add_argument("--out", type=Path, required=True, help="split json to write")
    p.add_argument("--sheet", type=Path, help="png of every boundary between splits and the most similar pairs")
    p.add_argument("--fractions", type=float, nargs=4, default=[0.4, 0.3, 0.1, 0.2], metavar=SPLITS,
                   help="share of the images and boxes for " + " / ".join(SPLITS))
    p.add_argument("--pool", type=int, default=4, help="cuts compare 64x48 thumbnails average-pooled by this")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--vis_cut", type=float, default=0.7)
    p.add_argument("--ir_cut", type=float, default=0.85)
    p.add_argument("--min_block", type=int, default=5, help="a block shorter than this is merged with both neighbours")
    p.add_argument("--vis_dup", type=float, default=0.99)
    p.add_argument("--ir_dup", type=float, default=0.99)
    p.add_argument("--verify_window", type=int, default=3, help="frames on each side of a cut checked with --same_*")
    p.add_argument("--same_either", type=float, default=0.85,
                   help="shift-tolerant correlation: same place if Vis or Ir reaches this")
    p.add_argument("--same_both", type=float, default=0.8, help="... or if both reach this")
    p.add_argument("--revisit_vis", type=float, default=0.8, help="revisit candidates: coarse Vis correlation above")
    p.add_argument("--revisit_ir", type=float, default=0.85, help="... and coarse Ir correlation above")
    p.add_argument("--revisit_same", type=float, default=0.88, help="revisit: shift-tolerant Vis and Ir at least this")
    p.add_argument("--revisit_pairs", type=int, default=5, help="... for at least this many pairs between two blocks")
    p.add_argument("--balance", nargs="*", default=[], choices=["light", "haze", "size"],
                   help="image conditions to balance across splits besides the classes (v2: light haze size)")
    p.add_argument("--balance_weight", type=float, default=3, help="weight of each --balance condition (classes: 1)")
    p.add_argument("--dark_below", type=float, default=100, help="night: mean Vis thumbnail below this (0-255)")
    p.add_argument("--haze_std_below", type=float, default=25, help="haze: not night, Vis thumbnail std below this")
    p.add_argument("--guard", type=int, default=2, help="images on each side of a boundary between splits left out")
    p.add_argument("--drop", nargs="*", default=["01300"],
                   help="images left out of every split: 01300 is a broken pair (its Vis is 00536, its Ir 00539)")
    p.add_argument("--d5_classes", nargs="*", default=["Bus", "Motorcycle", "Truck"],
                   help="classes to transport (PLAN D5), with box floors in rgb_train and test")
    p.add_argument("--d5_rgb_train", type=int, default=200)
    p.add_argument("--d5_test", type=int, default=50)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--date_created", help="date written in the json (default: today)")
    args = p.parse_args()
    if abs(sum(args.fractions) - 1) > 1e-6:
        p.error("--fractions must sum to 1")
    main(args)
