"""Prompt for the classifier.

The few-shot examples are written separately from data/appeals.csv on purpose:
using evaluation appeals as examples would leak the answers and inflate the score.
"""
from .labels import LABELS

FEW_SHOT = [
    ("I hold a disabled badge and the bays near the lift were full.", "blue_badge"),
    ("There is no sign at the barrier telling you it's pay and display.", "signage"),
    ("The machine took my money but the screen froze and no ticket came out.", "payment_error"),
    ("I'm staff and my permit was renewed but your system still shows it as expired.", "permit"),
    ("My daughter collapsed and we were in resus for four hours.", "emergency"),
    ("Someone else was driving my car that day.", "other"),
]


def system_prompt() -> str:
    defs = "\n".join(f"- {name}: {desc}" for name, desc in LABELS.items())
    examples = "\n".join(f'Appeal: "{t}"\nLabel: {l}' for t, l in FEW_SHOT)
    return f"""You classify private parking charge appeals for a UK car park operator.

Choose exactly ONE main reason for each appeal from these labels:
{defs}

Rules:
- Pick the reason the driver relies on most. If two apply, prefer the more specific one
  (for example a Blue Badge holder complaining about signs is blue_badge).
- Personal details have been replaced with [REG], [EMAIL] or [PHONE]; ignore them.
- confidence is your probability (0 to 1) that the label is correct.

Examples:
{examples}

Reply with JSON only, no other text:
{{"label": "<one label>", "confidence": <0-1>, "reason": "<max 15 words>"}}"""


def user_prompt(appeal_text: str) -> str:
    return f'Appeal: "{appeal_text}"'
