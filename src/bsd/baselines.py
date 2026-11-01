"""Re-run the CSE437 paper's classical models on the same splits as the transformer.

    python -m bsd.baselines
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import features, metrics
from .data import load_split

C_GRID = [0.1, 1.0, 10.0]


def models():
    """name -> (featurize fn, estimator factory taking C)."""
    raw = lambda texts: list(texts)  # noqa: E731
    return {
        "logreg_features": (
            features.matrix,
            lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)),
        ),
        "logreg_tfidf_word": (
            raw,
            lambda c: make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000),
                LogisticRegression(C=c, max_iter=2000),
            ),
        ),
        "linsvm_tfidf_char": (
            raw,
            lambda c: make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000),
                LinearSVC(C=c),
            ),
        ),
    }


def score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def run(data_dir: Path, out: Path) -> dict:
    splits = {name: load_split(name, data_dir) for name in ["train", "val", "test", "ood"]}
    y_train = splits["train"].label
    results = {
        "majority_baseline": {
            split: metrics.compute(splits[split].label, np.zeros(len(splits[split]), dtype=int))
            for split in ("test", "ood")
        }
    }
    for name, (featurize, factory) in models().items():
        x = {split: featurize(df.text) for split, df in splits.items()}

        # pick C on the validation split only, so test and OOD stay untouched
        def val_f1(c, featurized=x, make=factory):
            preds = make(c).fit(featurized["train"], y_train).predict(featurized["val"])
            return metrics.compute(splits["val"].label, preds)["f1"]

        best_c = max(C_GRID, key=val_f1)
        model = factory(best_c).fit(x["train"], y_train)
        results[name] = {"C": best_c}
        for split in ("test", "ood"):
            y, s = splits[split].label, score(model, x[split])
            results[name][split] = metrics.compute(y, model.predict(x[split]), s)
            metrics.save_scores(name, split, y, s)
        print(f"{name:20s} C={best_c:<5} test acc={results[name]['test']['accuracy']:.3f} "
              f"ood acc={results[name]['ood']['accuracy']:.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/baselines.json"))
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
