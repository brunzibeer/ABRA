"""``abra report``: collect ``results.json`` files of ``abra eval`` runs into one table (markdown + csv)."""
import csv
import json
from pathlib import Path

COLUMNS = ["run", "method", "alpha", "oracle", "AP", "AP50", "AP75"]


def add_report_args(p):
    p.add_argument("roots", nargs="+", type=Path, help="directories searched recursively for results.json")
    p.add_argument("--csv", type=Path, default=None, help="also write the table as csv")


def collect(roots):
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob("results.json")):
            r = json.loads(path.read_text())
            config = path.parent / "config.json"
            run = json.loads(config.read_text())["args"].get("exp_name") if config.exists() else path.parent.name
            rows.append({"run": run, "method": r.get("method"), "alpha": r.get("alpha"), "oracle": r.get("oracle"),
                         "AP": r.get("AP"), "AP50": r.get("AP50"), "AP75": r.get("AP75")})
    return rows


def report(args):
    rows = collect(args.roots)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for row in rows:
        print("| " + " | ".join(fmt(row[c]) for c in COLUMNS) + " |")
    if rows:
        print(f"\n{len(rows)} runs, mean AP {sum(r['AP'] for r in rows) / len(rows):.2f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return rows
