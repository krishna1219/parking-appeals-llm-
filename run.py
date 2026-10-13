"""Classify the appeals and score them.

    python run.py --classifier baseline           # no API key needed
    python run.py --classifier llm                # Anthropic (ANTHROPIC_API_KEY)
    python run.py --classifier llm --provider openai   # OPENAI_API_KEY + LLM_MODEL
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.baseline import KeywordClassifier
from src.classify import LLMClassifier
from src.evaluate import evaluate, print_summary

ROOT = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", choices=["baseline", "llm"], default="baseline")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default=None)
    p.add_argument("--data", default=ROOT / "data" / "appeals.csv")
    p.add_argument("--limit", type=int, default=None, help="only the first N appeals")
    p.add_argument("--delay", type=float, default=0,
                   help="seconds to wait between calls (free tiers have rate limits)")
    args = p.parse_args()

    load_dotenv()
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)

    if args.classifier == "baseline":
        clf, name = KeywordClassifier(), "baseline"
    else:
        clf, name = LLMClassifier(args.provider, args.model), f"llm_{args.provider}"

    preds = []
    for i, text in enumerate(df["appeal_text"], 1):
        preds.append(clf.predict(text).to_dict())
        if args.delay:
            time.sleep(args.delay)
        print(f"\r{name}: {i}/{len(df)}", end="", flush=True)

    out = pd.concat([df, pd.DataFrame(preds).add_prefix("pred_")], axis=1).rename(
        columns={"pred_needs_review": "needs_review", "pred_input_tokens": "input_tokens",
                 "pred_output_tokens": "output_tokens", "pred_latency_s": "latency_s"})
    metrics = evaluate(out, name, ROOT / "results")
    print_summary(metrics, out)


if __name__ == "__main__":
    main()
