"""Gradio demo: paste Bangla text, see whether it reads as LLM-generated or machine-translated.

    python app.py                      # uses models/banglabert
    MODEL_PATH=user/repo python app.py # or a model on the Hugging Face Hub
"""

import os

import gradio as gr
import pandas as pd

from bsd import features
from bsd.clean import normalize
from bsd.predict import Detector

detector = Detector(os.environ.get("MODEL_PATH", "models/banglabert"))

EXAMPLES = [
    "একদিন, পাহাড়ি গ্রামের একটি ছোট্ট ছেলে ছিল, নাম মনি। মনি খুব দুষ্টু ছিল, কিন্তু তার মনে অনেক কল্পনা ছিল। "
    "সে প্রতিদিন নতুন নতুন জিনিস আবিষ্কার করতে চাইত।",
    "একসময়, লিলি নামে একটি ছোট্ট মেয়ে ছিল। লিলি এমন এক কঠিন পৃথিবীতে বাস করত যেখানে সবকিছু করাই কঠিন ছিল। "
    "একদিন, লিলি স্কুল থেকে বাড়ি ফিরছিল।",
]


def classify(text: str):
    if not text.strip():
        return {}, pd.DataFrame()
    probs = detector.predict([text])[0]
    feats = features.extract(normalize(text))
    table = pd.DataFrame({"feature": list(feats), "value": [round(v, 3) for v in feats.values()]})
    return {"LLM-generated": probs["synthetic"], "Machine-translated": probs["translated"]}, table


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=8, label="Bangla text"),
    outputs=[gr.Label(label="Prediction"), gr.Dataframe(label="Interpretable features (from the paper)")],
    examples=EXAMPLES,
    title="Bangla Source Detector",
    description="Fine-tuned BanglaBERT that tells LLM-generated Bangla from machine-translated Bangla. "
    "Trained on Regional-TinyStories; accuracy drops on very different text, see the README.",
)

if __name__ == "__main__":
    demo.launch()
