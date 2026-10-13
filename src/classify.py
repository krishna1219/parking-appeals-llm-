"""LLM classifier: redact -> prompt -> call API -> validate JSON -> flag for review."""
import json
import os
import re
import time
from dataclasses import dataclass, asdict

from .labels import LABEL_NAMES
from .prompt import system_prompt, user_prompt
from .redact import redact

REVIEW_THRESHOLD = 0.7  # below this, a person checks the decision


@dataclass
class Prediction:
    label: str
    confidence: float
    reason: str
    needs_review: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""

    def to_dict(self):
        return asdict(self)


def parse_response(raw: str) -> dict:
    """Pull the JSON object out of the reply and check it is a valid label."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in reply: {raw[:80]!r}")
    data = json.loads(match.group(0))
    label = str(data.get("label", "")).strip().lower()
    if label not in LABEL_NAMES:
        raise ValueError(f"unknown label {label!r}")
    conf = float(data.get("confidence", 0))
    return {"label": label, "confidence": max(0.0, min(conf, 1.0)),
            "reason": str(data.get("reason", ""))[:200]}


class AnthropicBackend:
    def __init__(self, model=None):
        import anthropic  # imported here so the baseline runs without the SDK
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")

    def complete(self, system, user):
        r = self.client.messages.create(
            model=self.model, max_tokens=150, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens


class OpenAIBackend:
    def __init__(self, model=None):
        from openai import OpenAI
        self.client = OpenAI(timeout=60, max_retries=1)  # reads OPENAI_API_KEY
        self.model = model or os.environ["LLM_MODEL"]  # set to a current model name

    def complete(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=150,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        u = r.usage
        return r.choices[0].message.content, u.prompt_tokens, u.completion_tokens


BACKENDS = {"anthropic": AnthropicBackend, "openai": OpenAIBackend}


class LLMClassifier:
    def __init__(self, provider="anthropic", model=None, retries=2, backend=None):
        # `backend` lets tests pass a fake instead of calling a real API
        self.backend = backend or BACKENDS[provider](model)
        self.system = system_prompt()
        self.retries = retries

    def predict(self, appeal_text: str) -> Prediction:
        clean = redact(appeal_text)
        last_err = ""
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                raw, tin, tout = self.backend.complete(self.system, user_prompt(clean))
                p = parse_response(raw)
                return Prediction(**p, needs_review=p["confidence"] < REVIEW_THRESHOLD,
                                  input_tokens=tin, output_tokens=tout,
                                  latency_s=round(time.perf_counter() - start, 2))
            except Exception as e:  # bad JSON, rate limit, network
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)  # back off before retrying
        # never guess silently: failures go to a person
        return Prediction("other", 0.0, "", True, error=last_err)
