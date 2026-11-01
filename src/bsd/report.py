"""Collect every results/*.json into one comparison table (results/RESULTS.md) and chart.

    python -m bsd.report
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = ["accuracy", "f1", "macro_f1", "roc_auc", "recall_synthetic", "recall_translated"]


def collect(results_dir: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "test" in data:  # a single transformer run
            rows[path.stem] = data
        else:  # baselines.json holds several models; other files (calibration.json) are skipped
            rows.update({k: v for k, v in data.items() if isinstance(v, dict) and "test" in v})
    return rows


def table(rows: dict[str, dict], split: str) -> str:
    lines = ["| Model | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    ordered = sorted(rows.items(), key=lambda kv: kv[1][split]["accuracy"], reverse=True)
    for name, r in ordered:
        cells = [f"{r[split].get(c, float('nan')):.3f}" for c in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def chart(rows: dict[str, dict], out: Path) -> None:
    names = sorted(rows, key=lambda n: rows[n]["ood"]["accuracy"])
    y = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.2))
    ax.barh([i + 0.2 for i in y], [rows[n]["test"]["accuracy"] for n in names], 0.4,
            label="In-domain test", color="#4c78a8")
    ax.barh([i - 0.2 for i in y], [rows[n]["ood"]["accuracy"] for n in names], 0.4,
            label="Out-of-domain", color="#f58518")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(list(y), names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    results_dir = Path("results")
    rows = collect(results_dir)
    md = ["# Results", "", "## In-domain test (Regional-TinyStories held-out)", "", table(rows, "test"), "",
          "## Out-of-domain (Bangla-Instruct vs alpaca-bengali)", "", table(rows, "ood"), ""]
    (results_dir / "RESULTS.md").write_text("\n".join(md))
    chart(rows, results_dir / "accuracy.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
