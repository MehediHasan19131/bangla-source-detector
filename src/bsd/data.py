"""Build the in-domain (Regional-TinyStories) and out-of-domain datasets.

The two TinyStories files are single 4-5 GB JSON arrays, so instead of downloading them we read
random 1 MB byte ranges over HTTP and parse the complete records inside each window. This gives an
unbiased sample spread over the whole file for a few hundred MB of traffic.

    python -m bsd.data --per-class 25000
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .clean import clean_texts

HF = "https://huggingface.co/datasets"
SYNTHETIC_URL = f"{HF}/TinyStories-Regional/beng-generated_4o-mini_2M/resolve/main/train.json"
TRANSLATED_URL = (
    f"{HF}/TinyStories-Regional/beng-translated_googletrans_1.8M/resolve/main/"
    "orig-tiny_stories-translated/data.json"
)
RECORD = re.compile(r'\{\s*"story":\s*"((?:[^"\\]|\\.)*)",\s*"ID":\s*"([^"]*)"\s*\}')
WINDOW = 1_000_000


def file_size(url: str) -> int:
    resp = requests.head(url, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def parse_window(chunk: bytes) -> list[tuple[str, str]]:
    """Return (id, story) pairs for every record that sits fully inside the byte window."""
    text = chunk.decode("utf-8", errors="ignore")
    return [(m.group(2), json.loads(f'"{m.group(1)}"')) for m in RECORD.finditer(text)]


def range_sample(url: str, n: int, seed: int) -> list[str]:
    """Sample at least `n` distinct stories from random windows of a large JSON-array file."""
    rng = random.Random(seed)
    size = file_size(url)
    stories: dict[str, str] = {}
    while len(stories) < n:
        start = rng.randrange(0, size - WINDOW)
        resp = requests.get(url, headers={"Range": f"bytes={start}-{start + WINDOW - 1}"}, timeout=60)
        resp.raise_for_status()
        stories.update(parse_window(resp.content))
        print(f"  {url.split('/')[5]}: {min(len(stories), n)}/{n}", end="\r")
    print()
    return list(stories.values())


def load_ood(seed: int, rows: int = 40_000) -> tuple[list[str], list[str]]:
    """Out-of-domain test texts from other sources (same pairing as the paper).

    synthetic: responses written natively in Bangla by an LLM (Bangla-Instruct)
    translated: Google-translated Alpaca outputs (alpaca-bengali-cleaned)
    """
    from datasets import concatenate_datasets, load_dataset

    instruct = load_dataset("md-nishat-008/Bangla-Instruct", split="train", streaming=True)
    synthetic = [row["response"] for row in instruct.shuffle(seed=seed, buffer_size=10_000).take(rows)]
    alpaca = load_dataset("saillab/alpaca-bengali-cleaned")
    translated = concatenate_datasets([alpaca["train"], alpaca["test"]])["output"]
    return synthetic, translated


def balanced(synthetic: list[str], translated: list[str], n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(synthetic), len(translated))
    rng = random.Random(seed)
    rows = [(t, 0) for t in rng.sample(synthetic, n)] + [(t, 1) for t in rng.sample(translated, n)]
    return pd.DataFrame(rows, columns=["text", "label"])


def build(out_dir: Path, per_class: int, ood_per_class: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    oversample = int(per_class * 1.1)  # headroom for rows removed by cleaning

    print("Sampling Regional-TinyStories ...")
    synthetic = clean_texts(range_sample(SYNTHETIC_URL, oversample, seed))
    translated = clean_texts(range_sample(TRANSLATED_URL, oversample, seed + 1))
    domain = balanced(synthetic, translated, per_class, seed)

    train, rest = train_test_split(domain, test_size=0.2, stratify=domain.label, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=seed)

    print("Loading out-of-domain sources ...")
    ood_syn, ood_tr = load_ood(seed)
    ood = balanced(clean_texts(ood_syn), clean_texts(ood_tr), ood_per_class, seed)
    ood = ood[~ood.text.isin(set(domain.text))]

    stats = {}
    for name, df in {"train": train, "val": val, "test": test, "ood": ood}.items():
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        stats[name] = {"rows": len(df), **{f"label_{k}": int(v) for k, v in df.label.value_counts().items()}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def load_split(name: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=25_000, help="in-domain stories per class")
    parser.add_argument("--ood-per-class", type=int, default=3_580, help="out-of-domain texts per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.out, args.per_class, args.ood_per_class, args.seed)


if __name__ == "__main__":
    main()
