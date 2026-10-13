"""Keyword baseline. The LLM has to beat this to justify its cost.

The keywords come ONLY from the label definitions in labels.py, written before
looking at the evaluation appeals. Tuning keywords on the test set gave a fake
96% in an earlier version - a good example of test-set leakage.

It needs no API key, so the whole pipeline can be run and tested offline.
"""
from .classify import Prediction

# Checked in the order the labels are defined: the first rule that matches wins.
RULES = [
    ("blue_badge", ["blue badge", "disabled badge", "accessible bay"]),
    ("signage", ["sign"]),
    ("payment_error", ["paid", "pay", "payment", "machine", "app "]),
    ("permit", ["permit"]),
    ("emergency", ["emergency", "breakdown", "accident", "evacuat", "bereave", "medical"]),
]


class KeywordClassifier:
    def predict(self, appeal_text: str) -> Prediction:
        text = appeal_text.lower()
        for label, words in RULES:
            if any(w in text for w in words):
                return Prediction(label, 1.0, "keyword match", False)
        return Prediction("other", 0.5, "no keyword matched", True)
