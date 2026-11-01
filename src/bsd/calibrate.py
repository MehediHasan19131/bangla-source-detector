"""Few-shot threshold calibration on the out-of-domain set.

Out of domain, the models still *rank* texts better than chance (ROC-AUC well above 0.5) but their
decision threshold no longer fits, so nearly everything is labelled "translated". This asks: if an
auditor hand-labels k texts per class from the new source, how much accuracy does re-picking the
threshold on those k recover? Repeated over many random draws; the k labelled rows are never scored.

    python -m bsd.calibrate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHOTS = [10, 25, 50, 100]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing accuracy, placed midway between neighbouring scores."""
    order = np.sort(np.unique(scores))
    candidates = np.concatenate([[order[0] - 1e-9], (order[:-1] + order[1:]) / 2, [order[-1] + 1e-9]])
    accuracy = [((scores >= t).astype(int) == labels).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def run_model(df: pd.DataFrame, shots: list[int], repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k in shots:
        accs, synth_recalls = [], []
        for _ in range(repeats):
            calib = np.concatenate([rng.choice(np.flatnonzero(df.label == c), k, replace=False) for c in (0, 1)])
            rest = df.drop(index=calib)
            t = best_threshold(df.label.to_numpy()[calib], df.score.to_numpy()[calib])
            pred = (rest.score.to_numpy() >= t).astype(int)
            accs.append((pred == rest.label.to_numpy()).mean())
            synth_recalls.append((pred[rest.label.to_numpy() == 0] == 0).mean())
        out[k] = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "recall_synthetic_mean": round(float(np.mean(synth_recalls)), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=Path("results/scores"))
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {}
    for path in sorted(args.scores.glob("*__ood.parquet")):
        name = path.name.removesuffix("__ood.parquet")
        results[name] = run_model(pd.read_parquet(path), SHOTS, args.repeats, args.seed)

    lines = ["| Model | " + " | ".join(f"{k}/class" for k in SHOTS) + " |", "|---" * (len(SHOTS) + 1) + "|"]
    for name, by_k in sorted(results.items(), key=lambda kv: -kv[1][SHOTS[-1]]["accuracy_mean"]):
        cells = [f"{by_k[k]['accuracy_mean']:.3f} ± {by_k[k]['accuracy_std']:.3f}" for k in SHOTS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    Path("results/calibration.json").write_text(json.dumps(results, indent=2))
    Path("results/CALIBRATION.md").write_text(
        "# Out-of-domain accuracy after few-shot threshold calibration\n\n" + table + "\n"
    )
    print(table)


if __name__ == "__main__":
    main()
