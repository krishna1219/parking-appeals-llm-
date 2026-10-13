"""Score predictions against the hand labels and write the results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from .labels import LABEL_NAMES


def evaluate(df: pd.DataFrame, name: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = df["true_label"], df["pred_label"]

    metrics = {
        "classifier": name,
        "n": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABEL_NAMES,
                                   average="macro", zero_division=0), 3),
        "flagged_for_review": int(df["needs_review"].sum()),
        "per_class": classification_report(y_true, y_pred, labels=LABEL_NAMES,
                                           output_dict=True, zero_division=0),
    }
    reviewed = df[~df["needs_review"]]
    metrics["accuracy_on_auto_decided"] = (
        round(accuracy_score(reviewed["true_label"], reviewed["pred_label"]), 3)
        if len(reviewed) else None)
    if "input_tokens" in df and df["input_tokens"].sum():
        metrics["total_input_tokens"] = int(df["input_tokens"].sum())
        metrics["total_output_tokens"] = int(df["output_tokens"].sum())
        metrics["mean_latency_s"] = round(df["latency_s"].mean(), 2)

    df.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_confusion(y_true, y_pred, name, out_dir / f"{name}_confusion.png")
    return metrics


def _plot_confusion(y_true, y_pred, name, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (hand label)")
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(m: dict, df: pd.DataFrame):
    print(f"\n=== {m['classifier']} ===")
    print(f"Accuracy {m['accuracy']:.0%} | macro F1 {m['macro_f1']:.2f} | "
          f"flagged for review {m['flagged_for_review']}/{m['n']}")
    if m.get("accuracy_on_auto_decided") is not None:
        print(f"Accuracy on appeals decided without review: {m['accuracy_on_auto_decided']:.0%}")
    if m.get("total_input_tokens"):
        print(f"Tokens in/out: {m['total_input_tokens']}/{m['total_output_tokens']} | "
              f"mean latency {m['mean_latency_s']}s")
    wrong = df[df["true_label"] != df["pred_label"]]
    if len(wrong):
        print("Misclassified:")
        for _, r in wrong.iterrows():
            print(f"  {r['appeal_id']}: true={r['true_label']} pred={r['pred_label']} "
                  f"— {r['appeal_text'][:70]}...")
