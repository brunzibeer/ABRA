"""Fig. 4 of the paper (domain expert ablation), from the revision CSV files only (revision round 2, T1).

    python tools/plot_fig4.py --round1 out/results.csv --round2 out_round2/results_round2.csv \
        --out images/domain_experts.pdf [--png images/domain_experts.png] [--two_bar]

Bars = averages over the five unavailable classes (Cityscapes -> Foggy Cityscapes, per-class protocol) of mAP and
AP50: "Zero shot w/ Object" (round 2, method ``--variant``), "Supervised" (round 1, E10 ``fig4_supervised``) and
ABRA (round 1, E1 ``abra``, the Table 1 row). ``--two_bar`` drops the first bar. Style of the original
``images/domain_experts.pdf`` (351.3 x 170.3 pt, DejaVu Sans embedded as TrueType). Deterministic output.
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["bus", "motorcycle", "rider", "train", "truck"]
BARS = [  # (label, csv, method, fill, value-label color, hatch)
    ("Zero shot w/ Object", "round2", None, "#9481BB", "#3D3060", "/"),
    ("Supervised", "round1", "fig4_supervised", "#D3D3D3", "#5A6270", None),
    ("ABRA", "round1", "abra", "#F9EB53", "#8A6E00", None),
]
SIZE_PT = (351.3, 170.3)


def averages(path, method):
    """(mean mAP, mean AP50) over CLASSES of the per-class Cityscapes -> Foggy rows of ``method``."""
    rows = {r["class"]: r for r in csv.DictReader(open(path))
            if (r["benchmark"], r["target_domain"], r["method"], r["protocol"]) == ("cityscapes", "fog", method, "per_class")}
    missing = [c for c in CLASSES if c not in rows]
    if missing:
        raise SystemExit(f"{path}: no {method} rows for {missing}")
    return tuple(statistics.mean(float(rows[c][m]) for c in CLASSES) for m in ("mAP", "AP50"))


def plot(bars, out, png=None):
    """Layout measured on the original PDF: axes box, expanded legend, font sizes (ticks 11, values 10 bold, legend 9)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 1.0})
    fig = plt.figure(figsize=(SIZE_PT[0] / 72, SIZE_PT[1] / 72))
    ax = fig.add_axes([0.1207, 0.1431, 0.860, 0.7743])
    width = 0.2
    offsets = [(i - (len(bars) - 1) / 2) * width for i in range(len(bars))]
    for (label, values, fill, text_color, hatch), dx in zip(bars, offsets):
        xs = [g + dx for g in range(2)]
        ax.bar(xs, values, width, color=fill, edgecolor="black", linewidth=1, hatch=hatch, label=label, zorder=3)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=text_color, zorder=4)
    ax.set_xticks(range(2), ["mAP", "AP$_{50}$"], fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(30, 70)
    ax.set_yticks(range(30, 71, 10))
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("%", rotation=0, fontsize=11)
    ax.yaxis.set_label_coords(-0.105, 0.5)
    ax.yaxis.grid(True, linestyle=(0, (2.59, 1.12)), color="#808080", linewidth=0.7, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0316, 0.922, 0.936, 0.1276), mode="expand", ncol=len(bars),
                       fontsize=9, frameon=True, fancybox=False, borderaxespad=0)
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    if png:
        fig.savefig(png, dpi=200, metadata={"Software": None})
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round1", type=Path, required=True, help="round-1 results.csv (Supervised, ABRA)")
    p.add_argument("--round2", type=Path, default=None, help="results_round2.csv (Zero shot w/ Object)")
    p.add_argument("--variant", default="fig4_zs_obj", help="method of the Zero shot w/ Object bar in --round2")
    p.add_argument("--two_bar", action="store_true", help="Supervised and ABRA only")
    p.add_argument("--out", type=Path, required=True, help="PDF")
    p.add_argument("--png", type=Path, default=None, help="200-dpi preview")
    args = p.parse_args(argv)
    paths = {"round1": args.round1, "round2": args.round2}
    bars = []
    for label, source, method, fill, text_color, hatch in BARS[1:] if args.two_bar else BARS:
        if paths[source] is None:
            raise SystemExit(f"--{source} is needed for the {label} bar (or use --two_bar)")
        values = averages(paths[source], method or args.variant)
        bars.append((label, values, fill, text_color, hatch))
        print(f"{label}: mAP {values[0]:.1f}, AP50 {values[1]:.1f}")
    plot(bars, args.out, args.png)


if __name__ == "__main__":
    main()
