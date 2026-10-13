import pytest

from src.classify import LLMClassifier, parse_response
from src.prompt import FEW_SHOT
from src.redact import redact

import pandas as pd
from pathlib import Path


def test_redacts_plate_email_phone():
    out = redact("Car AB12 CDE, email j.patel@example.com, call 07700 900123")
    assert "AB12" not in out and "@" not in out and "07700" not in out
    assert out.count("[REG]") == 1 and "[EMAIL]" in out and "[PHONE]" in out


def test_parse_valid_json_with_extra_text():
    p = parse_response('Sure: {"label": "Permit", "confidence": 0.92, "reason": "staff permit"}')
    assert p["label"] == "permit" and p["confidence"] == 0.92


def test_parse_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_response('{"label": "fraud", "confidence": 0.9}')


class FakeBackend:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def complete(self, system, user):
        self.seen.append(user)
        return self.reply, 400, 30


def test_low_confidence_goes_to_review_and_pii_never_sent():
    fake = FakeBackend('{"label": "signage", "confidence": 0.55, "reason": "unclear"}')
    pred = LLMClassifier(backend=fake).predict("No signs. My car is AB12 CDE.")
    assert pred.label == "signage" and pred.needs_review
    assert "AB12" not in fake.seen[0]


def test_bad_reply_fails_safe_to_review():
    pred = LLMClassifier(backend=FakeBackend("not json"), retries=0).predict("text")
    assert pred.needs_review and pred.error


def test_few_shot_examples_not_in_eval_set():
    data = pd.read_csv(Path(__file__).parents[1] / "data" / "appeals.csv")
    eval_texts = set(data["appeal_text"].str.lower())
    assert not any(t.lower() in eval_texts for t, _ in FEW_SHOT)
