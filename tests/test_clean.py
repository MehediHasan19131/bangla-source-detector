from bsd.clean import bangla_ratio, clean_texts, normalize

STORY = "একদিন একটি ছোট্ট মেয়ে বাগানে খেলতে গেল। সে একটি সুন্দর প্রজাপতি দেখল। " * 3


def test_normalize_collapses_whitespace():
    assert normalize("  এক\n\nদুই\tতিন  ") == "এক দুই তিন"


def test_normalize_applies_nfc():
    decomposed = "ো"  # e-kar + aa-kar composes to o-kar
    assert normalize(decomposed) == "ো"


def test_bangla_ratio():
    assert bangla_ratio("আমি") == 1.0
    assert bangla_ratio("abc") == 0.0
    assert bangla_ratio("") == 0.0
    assert bangla_ratio("আমি ab") == 0.6


def test_clean_texts_filters_and_dedupes():
    english = "Once upon a time there was a little girl who loved to play outside. " * 3
    out = clean_texts([STORY, STORY, english, "", "ছোট", None])
    assert out == [normalize(STORY)]


def test_clean_texts_length_bounds():
    assert clean_texts([STORY], max_chars=50) == []
    assert clean_texts([STORY], min_chars=10, max_chars=10_000) == [normalize(STORY)]
