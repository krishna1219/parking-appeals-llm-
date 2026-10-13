# Parking Appeals Classifier (LLM vs keyword baseline)

Classifies private parking charge appeals into six reasons with a Large Language Model,
and measures whether it actually beats a simple keyword baseline.

**Why:** UK car park operators receive large volumes of appeals that are read and routed by hand.
Tagging the main reason automatically speeds up triage, while low-confidence cases still go to a person.

| Label | Example |
| --- | --- |
| `blue_badge` | Badge displayed but slipped off the dashboard |
| `signage` | Signs hidden, unlit or contradictory |
| `payment_error` | Paid in the app but no payment recorded |
| `permit` | Valid resident/staff permit not linked to a new car |
| `emergency` | Drove straight to A&E; breakdown; evacuation |
| `other` | Not the driver, car sold, cloned plates, grace period |

## How it works

```
appeal text ─► redact PII ─► prompt (definitions + few-shot) ─► LLM ─► validate JSON
                 (reg, email,                                          │
                  phone)                          confidence < 0.7 or error ─► human review
```

- **Data:** `data/appeals.csv` — 50 synthetic appeals across hospital, residential, retail,
  university and commercial sites, hand-labelled (8–9 per class). No real people.
- **Privacy:** number plates, emails and phone numbers are replaced with `[REG]`, `[EMAIL]`,
  `[PHONE]` before anything leaves the machine (UK GDPR data minimisation).
- **Prompt:** clear, mutually exclusive label definitions plus six few-shot examples written
  separately from the test set, so the score isn't inflated by leakage (a test checks this).
- **Safe failure:** temperature 0, JSON validated against the label list, retries with backoff;
  bad replies or confidence below 0.7 are flagged for a person, never guessed.
- **Evaluation:** accuracy, macro F1, per-class precision/recall, confusion matrix, accuracy on
  the cases decided without review, tokens and latency.

## Results

| Classifier | Accuracy | Macro F1 | Sent to review |
| --- | --- | --- | --- |
| Keyword baseline | 76% | 0.73 | 18 / 50 |
| LLM | *run `python run.py --classifier llm` and add your numbers here* | | |

The baseline's main weakness shows in `results/baseline_confusion.png`: emergencies described
in everyday words ("went into labour", "flat tyre", "passed away") fall through to `other`,
and "Paying for parking was the last thing on my mind" is wrongly read as a payment error.
Understanding meaning rather than keywords is exactly where an LLM should add value.

**Lesson learned:** an earlier baseline scored 96% because its keywords were written after
reading the test appeals. That is test-set leakage. The current keywords come only from the
label definitions.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py --classifier baseline                 # no API key needed

cp .env.example .env                                # then add your key (free Gemini option inside)
python run.py --classifier llm --provider openai --delay 7 --limit 5   # Gemini free tier
python run.py --classifier llm --provider openai --delay 7             # all 50 appeals

pytest                                              # 6 tests, no API key needed
```

Outputs go to `results/`: predictions CSV, metrics JSON and confusion matrix PNG per classifier.

## Next steps

- Grow the test set with real (anonymised) appeals and measure label agreement between two people
- Try structured outputs / tool calling instead of parsing JSON from text
- Multi-label output, since some appeals give two reasons
- Route `blue_badge` and `emergency` appeals to priority review
- Track cost per 1,000 appeals and compare a small vs large model

## Project layout

```
data/appeals.csv        50 synthetic, hand-labelled appeals
src/labels.py           label definitions
src/redact.py           PII redaction
src/prompt.py           system prompt and few-shot examples
src/classify.py         LLM classifier (Anthropic or OpenAI), validation, review flag
src/baseline.py         keyword baseline
src/evaluate.py         metrics and confusion matrix
run.py                  command-line entry point
tests/test_pipeline.py  unit tests with a fake LLM backend
```
