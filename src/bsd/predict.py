"""Load a fine-tuned model and classify Bangla text.

    python -m bsd.predict "একদিন, একটি ছোট্ট মেয়ে ..."
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .clean import normalize


class Detector:
    def __init__(self, model_path: str = "models/banglabert", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        batch = self.tokenizer([normalize(t) for t in texts], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt")
        probs = self.model(**batch).logits.softmax(dim=-1)
        return [{"synthetic": p[0].item(), "translated": p[1].item()} for p in probs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--model", default="models/banglabert")
    args = parser.parse_args()
    print(Detector(args.model).predict([args.text])[0])


if __name__ == "__main__":
    main()
