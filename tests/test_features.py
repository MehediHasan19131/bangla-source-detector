import pytest

from bsd import features


def test_extract_returns_all_15_features():
    feats = features.extract("আমি ভাত খাই। তুমি কী খাও?")
    assert list(feats) == features.FEATURE_NAMES
    assert len(feats) == 15


def test_counts():
    feats = features.extract('সে বলল, "আমি যাব।" তারপর, সে গেল। কেন? ২০২৪ সাল.')
    assert feats["comma_count"] == 2
    assert feats["danda_count"] == 2
    assert feats["question_count"] == 1
    assert feats["period_count"] == 1
    assert feats["quote_count"] == 2
    assert feats["sentence_count"] == 4
    assert feats["digit_ratio"] > 0


def test_latin_and_bangla_ratio_are_complements_for_pure_text():
    feats = features.extract("hello আমি")
    assert feats["latin_ratio"] + feats["bangla_ratio"] == pytest.approx(1.0)


def test_empty_text_does_not_crash():
    feats = features.extract("")
    assert feats["char_len"] == 0
    assert feats["avg_word_len"] == 0


def test_matrix_shape():
    assert features.matrix(["আমি", "তুমি", "সে"]).shape == (3, 15)
