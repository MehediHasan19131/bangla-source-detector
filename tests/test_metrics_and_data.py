import json

import numpy as np

from bsd import metrics
from bsd.calibrate import best_threshold
from bsd.data import balanced, parse_window


def test_metrics_perfect_and_per_class_recall():
    m = metrics.compute([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_metrics_expose_one_class_collapse():
    # The paper's OOD failure: everything predicted "translated" looks fine on recall alone.
    m = metrics.compute([0, 0, 1, 1], [1, 1, 1, 1])
    assert m["recall_translated"] == 1.0
    assert m["recall_synthetic"] == 0.0
    assert m["accuracy"] == 0.5


def test_parse_window_skips_partial_records():
    records = [{"story": 'সে বলল, "হ্যাঁ।"\nশেষ।', "ID": "7"}, {"story": "দুই", "ID": "8"}]
    blob = json.dumps(records, ensure_ascii=False, indent=4).encode()
    assert parse_window(blob) == [("7", 'সে বলল, "হ্যাঁ।"\nশেষ।'), ("8", "দুই")]
    # cut in the middle of the first record and inside a multi-byte character
    assert parse_window(blob[25:]) == [("8", "দুই")]


def test_balanced_has_equal_classes():
    df = balanced(["a", "b", "c"], ["x", "y"], n=10, seed=0)
    assert df.label.value_counts().to_dict() == {0: 2, 1: 2}


def test_best_threshold_separates_shifted_scores():
    # every score sits above the default 0.5 cut-off, but the classes are still separable
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.90, 0.91, 0.92, 0.97, 0.98, 0.99])
    t = best_threshold(labels, scores)
    assert 0.92 < t < 0.97
