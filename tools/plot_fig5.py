"""Fig. 5 of the paper (effect of specialization), from the revision CSV files only (revision round 2, T2).

    python tools/plot_fig5.py --round1 out/results.csv --round2 out_round2/results_round2.csv \
        --out images/ensemble3.png [--pdf images/ensemble3.pdf]

Per class, AP50 of Merge (round 2, one expert trained on the five unavailable classes, transported with ABRA) and of
ABRA (round 1, E1 ``abra``, the Table 1 row), Cityscapes -> Foggy Cityscapes, per-class protocol. Style of the
original ``images/ensemble3.png``: 8 x 3 in, 1600 x 600 px at 200 dpi, DejaVu Sans (TrueType in the PDF).
Deterministic output.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["bus", "rider", "train", "motorcycle", "truck"]


def ap50(path, method):
    rows = {r["class"]: float(r["AP50"]) for r in csv.DictReader(open(path))
            if (r["benchmark"], r["target_domain"], r["method"], r["protocol"]) == ("cityscapes", "fog", method, "per_class")}
    missing = [c for c in ORDER if c not in rows]
    if missing:
        raise SystemExit(f"{path}: no {method} rows for {missing}")
    return [rows[c] for c in ORDER]


def plot(merge, abra, png, pdf=None):
    """Layout measured on the original PNG: axes box, bar width 0.3, default x margins, legend upper right."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 1.0})
    fig = plt.figure(figsize=(8.0, 3.0))
    ax = fig.add_axes([0.0884, 0.1458, 0.8914, 0.804])
    xs = range(len(ORDER))
    width = 0.3
    ax.bar([x - width / 2 for x in xs], merge, width, color="#9080C2", edgecolor="black", linewidth=1, hatch="/",
           label="Merge", zorder=3)
    ax.bar([x + width / 2 for x in xs], abra, width, color="#FAE455", edgecolor="black", linewidth=1, label="ABRA",
           zorder=3)
    ax.set_xticks(list(xs), [c.capitalize() for c in ORDER], fontsize=14)
    ax.set_ylim(0, 75)
    ax.set_yticks(range(0, 71, 10))
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("AP$_{50}$ (%)", fontsize=14)
    ax.yaxis.grid(True, linestyle="--", color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend = ax.legend(loc="upper right", ncol=2, fontsize=12, frameon=True, fancybox=False, framealpha=1)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=200, metadata={"Software": None})
    if pdf:
        fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round1", type=Path, required=True, help="round-1 results.csv (ABRA)")
    p.add_argument("--round2", type=Path, required=True, help="results_round2.csv (Merge)")
    p.add_argument("--merge", default="merge", help="method of the Merge bars in --round2")
    p.add_argument("--out", type=Path, required=True, help="PNG (200 dpi, 1600 x 600)")
    p.add_argument("--pdf", type=Path, default=None)
    args = p.parse_args(argv)
    merge, abra = ap50(args.round2, args.merge), ap50(args.round1, "abra")
    for c, m, a in zip(ORDER, merge, abra):
        print(f"{c:10s} Merge {m:6.2f}  ABRA {a:6.2f}")
    plot(merge, abra, args.out, args.pdf)


if __name__ == "__main__":
    main()
